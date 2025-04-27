import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from IPython.display import clear_output
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


lines_names = ['Train loss pred', 'Validate loss pred', 'Train loss time', 'Validate loss time']


def uniform_change_strategy(shape) -> torch.Tensor:
    return torch.rand(shape)


def scalar_dot_predictor(cls_result, from_emb, to_emb, **argkw) -> torch.Tensor:
    all_embs = torch.cat([from_emb, to_emb], dim=-1)
    # print(all_embs.shape, from_emb.shape, to_emb.shape, cls_result.shape)
    preds = (all_embs @ torch.unsqueeze(cls_result, 2)).squeeze(2)
    # print(preds.shape)
    return preds


def time_cross_predictior(cls_result, result, from_emb, to_emb, use_compositor=False, model=None) -> torch.Tensor:
    all_embs = torch.cat([from_emb, to_emb], dim=-1)
    # print(all_embs.shape, from_emb.shape, to_emb.shape, cls_result.shape, result.mT.shape)
    if use_compositor:
        # print(model.compositor.shape, from_emb.shape, to_emb.shape, result.shape)
        ones = torch.ones(*from_emb.shape[:2] , 1).to(from_emb.device)
        preds = torch.einsum("abc,sna,snb,snc->sn",
                             model.module.compositor if isinstance(model, nn.DataParallel) else model.compositor,
                             torch.concat([from_emb, ones], dim=2),
                             torch.concat([to_emb, ones], dim=2),
                             torch.concat([result, ones], dim=2),
                             )
    else:
        preds = (all_embs * result).sum(axis=-1)
    # print(preds.shape)
    return preds


def result_loss_slower_change(result, coef, **argkw):
    return ((result[:, :-1, :] - result[:, 1:, :]) ** 2).mean() * coef


def result_loss_empty(result, **argkw):
    return torch.Tensor([0]).to(result.device).mean()


def criterion_loss_fn(pred, target, msk_ind, save_ind, change_ind, criterion) -> torch.Tensor:
    ind = np.concatenate([msk_ind, change_ind, save_ind], axis=-1)
    return criterion(pred[:, ind], target[:, ind])


def scheduler_creator(optimizer, warmup_epochs, gamma, step_size):
    def lr_lambda(epoch):
        epoch //= step_size
        if epoch <= warmup_epochs:
            # print(f"Coef: {(epoch / warmup_epochs)}")
            return (epoch / warmup_epochs)
        else:
            # print(f"Coef: {(gamma ** (epoch - warmup_epochs))}")
            return (gamma ** (epoch - warmup_epochs))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def get_n_random_splits(data, probs):
    parts = np.cumsum(list(map(int, len(data) * np.array([*probs]))))
    idx = torch.randperm(len(data)).numpy()
    res = np.split(data[idx], parts[:-1])
    return res[:3]

def batch_to_model(batch, p_msk=0.15, p_change=0.15, p_save=0.15, device='cuda',
                   change_strategy=uniform_change_strategy):
    batch_size, cnt_words = batch['to_address'].shape

    # msk_or_change = np.random.choice(a=range(cnt_words), size=int(cnt_words * (p_change + p_msk)), replace=False)
    # msk_ind = np.random.choice(a=msk_or_change, size=int(cnt_words * p_msk))
    # change_ind = np.setdiff1d(msk_or_change, msk_ind)
    msk_ind, change_ind, save_ind = get_n_random_splits(np.arange(cnt_words), [p_msk, p_change, p_save])
    cnt_change = len(change_ind)

    volumes_features = batch['numeric_features']
    volumes_features[:, change_ind] = torch.unsqueeze(change_strategy((batch_size, cnt_change)), 2)
    
    batch_size, cnt_words = batch['to_address'].shape
    
    return msk_ind, change_ind, save_ind, dict(
        numeric_features=volumes_features.to(device),
        from_address=batch['from_address'].to(device),
        to_address=batch['to_address'].to(device),
        time_features=batch['time_features'].to(device),
        msk_ind=msk_ind + 1,
        volumes=volumes_features.to(device),
    )


def train_model(model, model_predictor, train_loader, val_loader, run, num_epochs=5, learning_rate=1e-5, loss_fn=criterion_loss_fn,
                p_change=0.15, p_msk=0.15, p_save=0.15, change_strategy=uniform_change_strategy, result_loss=result_loss_empty, device='cuda', start_epoch=0,
                warmup_epochs=5, gamma=0.8, step_size=4, seconds_betwen_image_show=10, time_coef_loss=1/128,  cnt_last_for_show=10,
                loggin_each=5, show_img=False, show_batch_size=1):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min',
                                                     factor=gamma,
                                                     patience=step_size,
                                                     threshold=0.01,
                                                     threshold_mode='abs',
                                                     min_lr=1e-10,
                                                     verbose=True)
    model.train()
    criterion = nn.MSELoss()
    all_train_losses = []
    all_val_losses = []
    all_train_loss_cross_time = []
    all_val_loss_cross_time = []
    last_time = pd.Timestamp.now()
    for epoch in range(start_epoch, num_epochs):

        train_loss = 0
        train_loss_cross_time = 0
        np.random.seed(None)
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            msk_ind, change_ind, save_ind, model_input = batch_to_model(
                batch=batch, p_msk=p_msk, p_change=p_change, p_save=p_save,
                device=device, change_strategy=change_strategy)
            res = model(
                **model_input
            )
            cls_result = res['cls_result']
            result = res['result']
            from_emb = res['from_emb']
            to_emb = res['to_emb']

            volumes_pred = model_predictor(
                cls_result=cls_result,
                result=result,
                from_emb=from_emb,
                to_emb=to_emb,
                use_compositor=model.module.use_compositor if isinstance(model, nn.DataParallel) else model.use_compositor,
                model=model,)
            loss = loss_fn(
                pred=volumes_pred,
                target=batch['value'].to(volumes_pred.device),
                msk_ind=msk_ind,
                change_ind=change_ind,
                criterion=criterion,
                save_ind=save_ind)
            train_loss += loss.cpu().detach().item()

            loss_cross_time = result_loss(result=result, coef=time_coef_loss).to(volumes_pred.device)
            loss += loss_cross_time

            train_loss_cross_time += loss_cross_time.cpu().detach().item()

            loss.backward()
            optimizer.step()
            
        np.random.seed(42)
        with torch.no_grad():
            val_loss = 0
            val_loss_cross_time = 0
            for batch in tqdm(val_loader):

                msk_ind, change_ind, save_ind, model_input = batch_to_model(
                    batch=batch, p_msk=p_msk, p_change=p_change, p_save=p_save,
                    device=device, change_strategy=change_strategy)
                res = model(
                    **model_input
                )
                cls_result = res['cls_result']
                result = res['result']
                from_emb = res['from_emb']
                to_emb = res['to_emb']

                volumes_pred = model_predictor(
                    cls_result=cls_result,
                    result=result,
                    from_emb=from_emb,
                    to_emb=to_emb,
                     use_compositor=model.module.use_compositor if isinstance(model, nn.DataParallel) else model.use_compositor,
                    model=model,)
                val_loss += loss_fn(
                    pred=volumes_pred,
                    target=batch['value'].to(volumes_pred.device),
                    msk_ind=msk_ind,
                    change_ind=change_ind,
                    criterion=criterion,
                    save_ind=save_ind,).cpu().detach().item()
                val_loss_cross_time += result_loss(
                    result=result, coef=time_coef_loss).cpu().detach().item()

        all_train_losses.append(train_loss / len(train_loader))
        all_val_losses.append(val_loss / len(val_loader))
        all_train_loss_cross_time.append(train_loss_cross_time / len(train_loader))
        all_val_loss_cross_time.append(val_loss_cross_time / len(val_loader))
        scheduler.step(val_loss + val_loss_cross_time)
        
        run.log(dict(
            train_loss=train_loss / len(train_loader),
            val_loss=val_loss / len(val_loader),
            train_loss_cross_time = train_loss_cross_time / len(train_loader),
            val_loss_cross_time = val_loss_cross_time / len(val_loader),
            
        ))

        if pd.Timestamp.now() - last_time > pd.Timedelta(seconds=seconds_betwen_image_show) or epoch % loggin_each == 0:
            need_draw=False
            if pd.Timestamp.now() - last_time > pd.Timedelta(seconds=seconds_betwen_image_show):
                last_time = pd.Timestamp.now()
                need_draw=True
            graphs_names = [
                "Все результаты",
                f"Последние {cnt_last_for_show} train loss",
                f"Последние {cnt_last_for_show} time loss",
                f"Gramm matrix of known adress embedings",
                f'Predicts vs target (val)',
            ]
            figs = {
                graphs_names[0]: go.Figure(),
                graphs_names[1]: go.Figure(),
                graphs_names[2]: go.Figure(),
                graphs_names[3]: go.Figure(),
                graphs_names[4]: go.Figure(),
            }
            clear_output()
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss / len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
            print(f'lr: {scheduler.get_last_lr()}')
            run.log(
                dict(epoch=epoch+1,
                     train_loss=train_loss / len(train_loader),
                     val_loss=val_loss / len(val_loader),
                     lr=scheduler.get_last_lr()))

            fig = make_subplots(rows=5, cols=1, subplot_titles=graphs_names)

            for trace in [
                go.Scatter(x=np.arange(epoch + 1), y=all_train_losses,
                           name=lines_names[0], mode='lines'),
                go.Scatter(x=np.arange(epoch + 1), y=all_val_losses,
                           name=lines_names[1], mode='lines'),
                go.Scatter(x=np.arange(epoch + 1), y=all_train_loss_cross_time,
                           name=lines_names[2], mode='lines'),
                go.Scatter(x=np.arange(epoch + 1), y=all_val_loss_cross_time,
                           name=lines_names[3], mode='lines'),
            ]:
                fig.add_trace(
                    trace,
                    row=1, col=1,
                )
                figs[graphs_names[0]].add_trace(trace)
                
            for trace in [
                go.Scatter(x=np.arange(epoch + 1), y=all_train_losses[-cnt_last_for_show:],
                               name=lines_names[0], mode='lines'),
                go.Scatter(x=np.arange(epoch + 1), y=all_val_losses[-cnt_last_for_show:],
                               name=lines_names[1], mode='lines')
            ]:
                fig.add_trace(trace, row=2, col=1)
                figs[graphs_names[1]].add_trace(trace)

            fig.update_layout(
                xaxis_title='epoch',
                yaxis_title='loss',
                width=1000,  height=1500,                 
            )

            for trace in [
                go.Scatter(x=np.arange(epoch + 1), y=all_train_loss_cross_time[-cnt_last_for_show:],
                               name=lines_names[2], mode='lines'),
                go.Scatter(x=np.arange(epoch + 1), y=all_val_loss_cross_time[-cnt_last_for_show:],
                                     name=lines_names[3], mode='lines')
            ]:
                fig.add_trace(trace, row=3, col=1)
                figs[graphs_names[2]].add_trace(trace)


            adress_emb = model.address_embedding._parameters['weight'].detach().cpu()
            trace = go.Heatmap(z=adress_emb @ adress_emb.T,
                               colorbar=dict(len=0.2, y=0.3, yanchor='middle', x=1.05))
            fig.add_trace(trace, row=4, col=1)
            figs[graphs_names[3]].add_trace(go.Heatmap(z=adress_emb @ adress_emb.T,))
            
            with torch.no_grad():
                raw_show_batch = []
                it = iter(train_loader)
                raw_show_batch.append(next(it))
                for _ in range(show_batch_size // raw_show_batch[0]['time_features'].numel()):
                    raw_show_batch.append(next(it))
                keys = raw_show_batch[0].keys()
                
                show_batch = dict()
                for key in keys:
                    show_batch[key] = torch.concat([raw[key] for raw in raw_show_batch])
                
                msk_ind, change_ind, save_ind, model_input = batch_to_model(
                    batch=show_batch, p_msk=p_msk, p_change=p_change, p_save=p_save,
                    device=device, change_strategy=change_strategy)
                res = model(
                    **model_input
                )
                cls_result = res['cls_result']
                result = res['result']
                from_emb = res['from_emb']
                to_emb = res['to_emb']

                volumes_pred = model_predictor(
                    cls_result=cls_result,
                    result=result,
                    from_emb=from_emb,
                    to_emb=to_emb,
                    use_compositor=model.module.use_compositor if isinstance(model, nn.DataParallel) else model.use_compositor,
                    model=model,)
                
                indexes = show_batch['time_features'][..., -1].detach().cpu().reshape(-1)
                # print(indexes.shape)
                indexes_sorted = sorted(np.arange(len(indexes)), key=lambda x: indexes[x,])
                indexes_sorted = indexes_sorted[:show_batch_size]

                targets = show_batch['numeric_features'][..., -1].detach().cpu().reshape(-1)
                preds = volumes_pred.reshape(-1).detach().cpu()

                for trace in [
                    go.Scatter(
                        x=np.arange(len(indexes_sorted)),
                        y=targets[indexes_sorted],
                        line_shape='hv',
                        name='Targets'),
                    go.Scatter(
                        x=np.arange(len(indexes_sorted)),
                        y=preds[indexes_sorted],
                        line_shape='hv',
                        name='predicts')
                ]:
                    fig.add_trace(trace, row=5, col=1,)
                    figs[graphs_names[4]].add_trace(trace)
                
            if epoch % loggin_each == 0:
                run.log(figs)
            if need_draw and show_img:
                fig.show()
