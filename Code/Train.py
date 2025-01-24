import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from IPython.display import clear_output

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def uniform_change_strategy(shape) -> torch.Tensor:
    return torch.rand(shape)


def scalar_dot_predictor(cls_result, result, from_emb, to_emb) -> torch.Tensor:
    all_embs = torch.cat([from_emb, to_emb], dim=-1)
    # print(all_embs.shape, from_emb.shape, to_emb.shape, cls_result.shape)
    return (all_embs @ torch.unsqueeze(cls_result, 2)).squeeze(2)


def criterion_loss_fn(pred, target, msk_ind, change_ind, criterion) -> torch.Tensor:
    ind = np.concatenate([msk_ind, change_ind], axis=-1)
    return criterion(pred[:, ind], target[:, ind])


def SchedulerCreator(optimizer, warmup_epochs, gamma, step_size):
    def lr_lambda(epoch):
        epoch //= step_size
        if epoch <= warmup_epochs:
            print(f"Coef: {(epoch / warmup_epochs)}")
            return (epoch / warmup_epochs)
        else:
            print(f"Coef: {(gamma ** (epoch - warmup_epochs))}")
            return (gamma ** (epoch - warmup_epochs))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_model(model, model_predictor, train_loader, val_loader, num_epochs=5, learning_rate=1e-5, loss_fn=criterion_loss_fn,
                p_change=0.15, p_msk=0.15, change_strategy=uniform_change_strategy, device='cuda', start_epoch=0,
                warmup_epochs=5, gamma=0.8, step_size=4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = SchedulerCreator(optimizer,
                                 warmup_epochs=warmup_epochs,
                                 gamma=gamma,
                                 step_size=step_size)
    model.train()
    criterion = nn.MSELoss()
    all_train_losses = []
    all_val_losses = []
    for epoch in range(start_epoch, num_epochs):
        scheduler.step(epoch)

        train_loss = 0
        np.random.seed(None)
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            batch_size, cnt_words = batch['to_address'].shape

            msk_or_change = np.random.choice(a=range(cnt_words), size=int(cnt_words * (p_change + p_msk)), replace=False)
            msk_ind = np.random.choice(a=msk_or_change, size=int(cnt_words * p_msk))
            change_ind = np.setdiff1d(msk_or_change, msk_ind)
            cnt_change = len(change_ind)

            volumes_features = batch['numeric_features']
            volumes_features[:, change_ind] = torch.unsqueeze(change_strategy((batch_size, cnt_change)), 2)

            res = model(
                numeric_features=volumes_features.to(device),
                from_address=batch['from_address'].to(device),
                to_address=batch['to_address'].to(device),
                time_features=batch['time_features'].to(device),
                msk_ind=msk_ind + 1
            )
            cls_result = res['cls_result']
            result = res['result']
            from_emb = res['from_emb']
            to_emb = res['to_emb']

            volumes_pred = model_predictor(cls_result, result, from_emb, to_emb)
            loss = loss_fn(
                pred=volumes_pred,
                target=batch['value'].to(device),
                msk_ind=msk_ind,
                change_ind=change_ind,
                criterion=criterion)
            train_loss += loss.cpu().detach().item()
            loss.backward()
            optimizer.step()

        np.random.seed(42)
        with torch.no_grad():
            val_loss = 0
            for batch in tqdm(val_loader):
                batch_size, cnt_words = batch['to_address'].shape

                msk_or_change = np.random.choice(a=range(cnt_words), size=int(cnt_words * (p_change + p_msk)), replace=False)
                msk_ind = np.random.choice(a=msk_or_change, size=int(cnt_words * p_msk))
                change_ind = np.setdiff1d(msk_or_change, msk_ind)
                cnt_change = len(change_ind)

                volumes_features = batch['numeric_features']
                volumes_features[:, change_ind] = torch.unsqueeze(change_strategy((batch_size, cnt_change)), 2)

                res = model(
                    numeric_features=volumes_features.to(device),
                    from_address=batch['from_address'].to(device),
                    to_address=batch['to_address'].to(device),
                    time_features=batch['time_features'].to(device),
                    msk_ind=msk_ind + 1
                )
                cls_result = res['cls_result']
                result = res['result']
                from_emb = res['from_emb']
                to_emb = res['to_emb']

                volumes_pred = model_predictor(cls_result, result, from_emb, to_emb)
                loss = loss_fn(
                    pred=volumes_pred,
                    target=batch['value'].to(device),
                    msk_ind=msk_ind,
                    change_ind=change_ind,
                    criterion=criterion)
                val_loss += loss.cpu().detach().item()

        clear_output()
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Test Loss: {val_loss / len(val_loader):.4f}')
        all_train_losses.append(train_loss / len(train_loader))
        all_val_losses.append(val_loss / len(val_loader))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(epoch + 1), y=all_train_losses, name='Train loss', mode='lines'))
        fig.add_trace(go.Scatter(x=np.arange(epoch + 1), y=all_val_losses, name='Validate loss', mode='lines'))

        fig.update_layout(
            xaxis_title='epoch',
            yaxis_title='loss'
        )

        fig.show()
