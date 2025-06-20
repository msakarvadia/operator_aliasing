"""Train and test utility functions."""

from __future__ import annotations

import typing

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
    lr = train_args['lr']
    weight_decay = train_args['weight_decay']
    gamma = train_args['gamma']
    step_size = train_args['step_size']
    loss = train_args['loss']
    device = train_args['device']
    seed = train_args['seed']
    train_dataloader = train_args['train_dataloader']
    test_dataloaders = train_args['test_dataloaders']
    ckpt_path = train_args['ckpt_path']
    ckpt_freq = train_args['ckpt_freq']

    # training stats
    # train_stats = pd.DataFrame()

    # TODO(MS): test seeding!!
    seed_everything(seed)

    # set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    # train model
    for epoch in range(starting_epoch, epochs + 1):
        train_loss = 0.0
        for _step, (input_data, output_data) in enumerate(train_dataloader):
            input_batch = input_data.to(device)
            output_batch = output_data.to(device)
            optimizer.zero_grad()
            output_pred_batch = model(input_batch)
            loss_f = loss(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer.step()
            train_loss += loss_f.item()
        train_loss /= len(train_dataloader)
        scheduler.step()

        # test model
        test_dict = test_model(model, test_dataloaders, device)
        test_relative_l2 = test_dict

        # save train stats:
        # train_stats.loc[-1] = {"epoch":epoch,
        # "train_loss":train_loss} | test_dict

        if epoch % ckpt_freq == 0:
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.to('cpu').state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            save_ckpt(ckpt_path, ckpt_dict)
            print(
                '######### Epoch:',
                epoch,
                ' ######### Train Loss:',
                train_loss,
                ' ######### Relative L1 Test Norm:',
                test_relative_l2,
            )
            model.to(device)

    return model.to('cpu')


# TODO(MS): make test function
def test_model(
    model: torch.nn.Module,
    test_dataloaders: dict[str, torch.utils.data.Dataloader],
    device: torch.device,
) -> dict[str, float]:
    """Test model."""
    test_dict = {}
    with torch.no_grad():
        model.eval()
        for test_label, test_dataloader in test_dataloaders.items():
            test_relative_l2 = 0.0
            for _step, (input_data, output_data) in enumerate(test_dataloader):
                input_batch = input_data.to(device)
                output_batch = output_data.to(device)
                output_pred_batch = model(input_batch)
                loss_f = (
                    torch.mean(abs(output_pred_batch - output_batch))
                    / torch.mean(abs(output_batch))
                ) ** 0.5 * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(test_dataloader)
            test_dict[test_label] = test_relative_l2
    return test_dict
