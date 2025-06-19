"""Train and test utility functions."""

from __future__ import annotations

import typing

import torch
from torch.nn import Module
from torch.optim import AdamW

from operator_aliasing.utils import seed_everything


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
    test_dataloader = train_args['train_dataloader']
    freq_print = 5

    # TODO(MS): test seeding!!
    seed_everything(seed)

    # set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    # train model
    model = model.to(device)
    for epoch in range(epochs):
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
        with torch.no_grad():
            model.eval()
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

        if epoch % freq_print == 0:
            print(
                '######### Epoch:',
                epoch,
                ' ######### Train Loss:',
                train_loss,
                ' ######### Relative L1 Test Norm:',
                test_relative_l2,
            )

    return model.to('cpu')


# TODO(MS): make test function
# def test_model():

#    return
# TODO(MS): make ckpting function
