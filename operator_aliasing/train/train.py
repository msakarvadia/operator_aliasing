"""Train and test utility functions."""

from __future__ import annotations

import typing

import pandas as pd
import torch
from torch.nn import Module
from torch.optim import AdamW

from operator_aliasing.train.utils import load_latest_ckpt
from operator_aliasing.train.utils import save_ckpt

from ..utils import seed_everything


def train_model(**train_args: typing.Any) -> Module:
    """Train a model."""
    # set up trianing args
    model = train_args['model']
    epochs = train_args['epochs']
    # lr = train_args['lr']
    # weight_decay = train_args['weight_decay']
    gamma = train_args['gamma']
    step_size = train_args['step_size']
    loss = train_args['loss']
    device = train_args['device']
    seed = train_args['seed']
    train_dataloader = train_args['train_dataloader']
    test_dataloaders = train_args['test_dataloaders']
    ckpt_path = train_args['ckpt_path']
    # ckpt_freq = train_args['ckpt_freq']
    # train_type = train_args['train_type']
    initial_steps = train_args['initial_steps']

    # training stats
    columns = ['epoch', 'train_loss', *list(test_dataloaders.keys())]
    train_stats = pd.DataFrame(columns=columns)

    # TODO(MS): test seeding!!
    seed_everything(seed)

    # set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_args['lr'],
        weight_decay=train_args['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
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
        print(train_stats)

    # train model
    for epoch in range(starting_epoch, epochs + 1):
        train_loss = 0.0
        for _step, batch in enumerate(train_dataloader):
            input_batch = batch['x'].to(device)
            output_batch = batch['y'].to(device)
            optimizer.zero_grad()
            if train_args['train_type'] != 'autoregressive':
                # for Darcy flow
                output_pred_batch = model(input_batch)
                loss_f = loss(output_pred_batch, output_batch)
            else:
                # Autoregressive loop for NS and burgers
                loss_f = autoregressive_loop(
                    input_batch, output_batch, initial_steps, model, loss
                )

            loss_f.backward()
            optimizer.step()
            train_loss += loss_f.item()
        train_loss /= len(train_dataloader)
        scheduler.step()

        # test model
        test_dict = test_model(model, test_dataloaders, device, loss)
        test_relative_l2 = test_dict

        # save train stats:
        train_stats.loc[len(train_stats)] = {
            'epoch': epoch,
            'train_loss': train_loss,
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
                #' ######### Relative L1 Test Norm:',
                test_relative_l2,
            )
            model.to(device)

    return model.to('cpu')


def autoregressive_loop(
    input_batch: torch.Tensor,
    output_batch: torch.tensor,
    initial_steps: int,
    model: Module,
    loss: Module,
) -> int:
    """Autoregressive training loop for time-varying PDE training."""
    loss_f = 0
    # Initialize the prediction tensor
    pred = input_batch[:, :initial_steps, ...]
    t_train = output_batch.shape[1]  # number of time steps
    for t in range(initial_steps, t_train):
        # Extract target at current time step
        output_at_time_step = output_batch[:, t : t + 1, ...].squeeze()

        print(f'Step {t}')
        # Model run
        output_pred_batch = model(input_batch)
        print(f'{output_pred_batch.shape=}, {output_at_time_step.shape=}')

        # Loss calculation
        loss_f += loss(output_pred_batch, output_at_time_step)

        # Concatenate the prediction at current time step into the
        # prediction tensor
        pred = torch.cat((pred, output_pred_batch), 1)

        # Concatenate the prediction at the current
        # time step to be used as input for the next time step
        input_batch = torch.cat(
            (input_batch[:, 1:, ...], output_pred_batch), dim=1
        )
    return loss_f


def test_model(
    model: Module,
    test_dataloaders: dict[str, torch.utils.data.Dataloader],
    device: torch.device,
    loss: Module,
) -> dict[str, float]:
    """Test model."""
    test_dict = {}
    with torch.no_grad():
        model.eval()
        for test_label, test_dataloader in test_dataloaders.items():
            test_relative_l2 = 0.0
            for _step, batch in enumerate(test_dataloader):
                input_batch = batch['x'].to(device)
                output_batch = batch['y'].to(device)
                output_pred_batch = model(input_batch)
                loss_f = loss(output_pred_batch, output_batch)
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
