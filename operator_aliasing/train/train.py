"""Train and test utility functions."""

from __future__ import annotations

import time
import typing

import pandas as pd
import torch
from torch.nn import Module
from torch.optim import AdamW
from tqdm import tqdm

from operator_aliasing.train.utils import load_latest_ckpt
from operator_aliasing.train.utils import save_ckpt
from operator_aliasing.train.utils import setup_logger

from ..utils import seed_everything
from .pinn_losses import Loss


def train_model(**train_args: typing.Any) -> Module:
    """Train a model."""
    # set up trianing args
    model = train_args['model']
    epochs = train_args['epochs']
    loss = train_args['loss']
    device = train_args['device']
    train_dataloader = train_args['train_dataloader']
    test_dataloaders = train_args['test_dataloaders']
    ckpt_path = train_args['ckpt_path']
    initial_steps = train_args['initial_steps']

    # set up logging
    logger = setup_logger(ckpt_path)
    logger.info(f'Training args: {train_args}')

    train_stats = pd.DataFrame(
        columns=[
            'epoch',
            'train_loss',
            'train_time',
            *list(test_dataloaders.keys()),
        ]
    )

    # TODO(MS): test seeding!!
    seed_everything(train_args['seed'])

    # set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_args['lr'],
        weight_decay=train_args['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_args['step_size'], gamma=train_args['gamma']
    )
    starting_epoch = 0
    model = model.to(device)

    # load ckpt if it exists
    ckpt_dict = load_latest_ckpt(ckpt_path)
    if ckpt_dict:
        model.load_state_dict(ckpt_dict['model_state_dict'])
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt_dict['scheduler_state_dict'])
        starting_epoch = ckpt_dict['epoch'] + 1
        train_stats = ckpt_dict['train_stats']
        logger.info(
            f'Resuming from: {ckpt_path} \n {ckpt_dict=} \n {train_stats}'
        )

    # train model
    for epoch in range(starting_epoch, epochs + 1):
        train_loss = 0.0
        start_time = time.time()
        # timing to exclude potential data loading
        forward_back_time = 0.0
        for _step, batch in enumerate(tqdm(train_dataloader)):
            # NOTE(MS): must remove outer batch dim from dataloader
            # because we pre-batch data due to multi-res training
            input_batch = batch['x'][0].to(device)
            output_batch = batch['y'][0].to(device)
            batch['device'] = device
            # print(f"{input_batch.shape=}, {output_batch.shape=}")

            optimizer.zero_grad()
            start_fb_time = time.time()
            if initial_steps == 1:
                # for Darcy flow
                output_pred_batch = model(input_batch)
                loss_f = loss(
                    output_pred_batch, output_batch, model_input=input_batch
                )
            else:
                # Autoregressive loop for NS and burgers
                loss_f = autoregressive_loop(
                    initial_steps, model, loss, **batch
                )

            loss_f.backward()
            end_fb_time = time.time()
            forward_back_time += end_fb_time - start_fb_time
            optimizer.step()
            train_loss += loss_f.item()
        scheduler.step()
        end_time = time.time()

        # test model
        test_dict = test_model(
            model,
            test_dataloaders,
            device,
            initial_steps,
        )

        # save train stats:
        train_stats.loc[len(train_stats)] = {
            'epoch': epoch,
            'train_loss': train_loss / len(train_dataloader),
            'train_time': end_time - start_time,
            'train_time_no_data_load': forward_back_time,
        } | test_dict

        if epoch % train_args['ckpt_freq'] == 0:
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.to('cpu').state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_stats': train_stats,
            }
            save_ckpt(ckpt_path, ckpt_dict)
            print(
                '######### Epoch:',
                epoch,
                ' ######### Train Loss:',
                train_loss,
                ' ######### Test Loss:',
                test_dict,
            )
            model.to(device)

    return model.to('cpu')


def autoregressive_loop(
    initial_steps: int,
    model: Module,
    loss: Module,
    **batch: typing.Any,
) -> int:
    """Autoregressive training loop for time-varying PDE training."""
    # adapted from: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/train.py
    device = batch['device']
    input_batch = batch['x'][0].to(device)
    output_batch = batch['y'][0].to(device)

    img_size = input_batch.shape[-1]
    batch_size = input_batch.shape[0]
    loss_f = 0
    t_train = output_batch.shape[1]  # number of time steps
    shape: typing.Any = (batch_size, -1)
    for _dim in range(model.n_dim):
        shape += (img_size,)

    all_model_preds = []
    for _t in range(initial_steps, t_train):
        # Model run
        model_input = torch.reshape(input_batch, shape)
        output_pred_batch = model(model_input)
        all_model_preds.append(output_pred_batch)

        # Concatenate the prediction at the current
        # time step to be used as input for the next time step
        input_batch = torch.cat(
            (input_batch[:, 1:, ...], output_pred_batch.unsqueeze(dim=1)),
            dim=1,
        )

    # stack all model preds
    all_model_preds = torch.stack(all_model_preds, dim=1)
    output_at_last_n_steps = output_batch[:, initial_steps:, ...]
    # Loss calculation
    loss_f = loss(
        all_model_preds,
        output_at_last_n_steps,
        model_input=input_batch,
        **batch,
    )
    return loss_f


def test_model(
    model: Module,
    test_dataloaders: dict[str, torch.utils.data.Dataloader],
    device: torch.device,
    initial_steps: int,
) -> dict[str, float]:
    """Test model."""
    test_dict = {}
    loss = Loss('mse')
    with torch.no_grad():
        model.eval()
        for test_label, test_dataloader in test_dataloaders.items():
            test_relative_l2 = 0.0
            for _step, batch in enumerate(test_dataloader):
                # NOTE(MS): must remove outer batch dim from dataloader
                # because we pre-batch data due to multi-res training
                input_batch = batch['x'][0].to(device)
                output_batch = batch['y'][0].to(device)
                batch['device'] = device

                if initial_steps == 1:
                    output_pred_batch = model(input_batch)
                    loss_f = loss(
                        output_pred_batch,
                        output_batch,
                        model_input=input_batch,
                    )
                else:
                    loss_f = autoregressive_loop(
                        initial_steps,
                        model,
                        loss,
                        **batch,
                    )
                """
                loss_f = (
                    torch.mean(abs(output_pred_batch - output_batch))
                    / torch.mean(abs(output_batch))
                ) ** 0.5 * 100
                """
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(test_dataloader)
            test_dict[test_label] = test_relative_l2
    return test_dict
