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
import pandas as pd
from plotly.subplots import make_subplots

# =============================================================================
# 1. Предикторы (Predictors)
# =============================================================================

def scalar_dot_predictor(cls_result, from_emb, to_emb, **kwargs) -> torch.Tensor:
    """
    Предиктор, который конкатенирует эмбеддинги и делает матричное умножение с классовым вектором.
    """
    all_embs = torch.cat([from_emb, to_emb], dim=-1)
    preds = (all_embs @ torch.unsqueeze(cls_result, 2)).squeeze(2)
    return preds

def time_cross_predictor(cls_result, result, from_emb, to_emb, use_compositor=False, model=None, **kwargs) -> torch.Tensor:
    """
    Предиктор, который для расчёта предсказаний использует произведение эмбеддингов с учетом результата.
    При наличии use_compositor применяется специальная композиция.
    """
    all_embs = torch.cat([from_emb, to_emb], dim=-1)
    if use_compositor:
        ones = torch.ones(*from_emb.shape[:2], 1).to(from_emb.device)
        preds = torch.einsum(
            "abc,sna,snb,snc->sn",
            model.module.compositor if isinstance(model, nn.DataParallel) else model.compositor,
            torch.cat([from_emb, ones], dim=2),
            torch.cat([to_emb, ones], dim=2),
            torch.cat([result, ones], dim=2),
        )
    else:
        preds = (all_embs * result).sum(axis=-1)
    return preds

# =============================================================================
# 2. Функции потерь (Loss Functions)
# =============================================================================

def result_loss_slower_change(result, coef, **kwargs) -> torch.Tensor:
    """
    Потеря, рассчитывающая изменения между соседними элементами результата.
    """
    return ((result[:, :-1, :] - result[:, 1:, :]) ** 2).mean() * coef

def result_loss_empty(result, **kwargs) -> torch.Tensor:
    """
    Заглушка для расчёта потерь – возвращает нулевую потерю.
    """
    return torch.Tensor([0]).to(result.device).mean()

def criterion_loss_fn(pred, target, msk_ind, save_ind, change_ind, criterion) -> torch.Tensor:
    """
    Основная функция потерь – рассчитывает MSE потери по выбранным индексам.
    """
    ind = np.concatenate([msk_ind, change_ind, save_ind], axis=-1)
    return criterion(pred[:, ind], target[:, ind])

# =============================================================================
# 3. Стратегии изменения (Change Strategies)
# =============================================================================

def uniform_change_strategy(shape) -> torch.Tensor:
    """
    Генерация равномерного случайного распределения для изменения данных.
    """
    return torch.rand(shape)

# =============================================================================
# 4. Вспомогательные функции (Utilities)
# =============================================================================

def scheduler_creator(optimizer, warmup_epochs, gamma, step_size):
    """
    Создает scheduler на основе LambdaLR с warmup-стадией.
    """
    def lr_lambda(epoch):
        epoch //= step_size
        if epoch <= warmup_epochs:
            return (epoch / warmup_epochs)
        else:
            return (gamma ** (epoch - warmup_epochs))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def get_n_random_splits(data, probs):
    """
    Разбивает данные на три части согласно переданным долям.
    """
    parts = np.cumsum(list(map(int, len(data) * np.array([*probs]))))
    idx = torch.randperm(len(data)).numpy()
    res = np.split(data[idx], parts[:-1])
    return res[:3]

def batch_to_model(batch, p_msk=0.15, p_change=0.15, p_save=0.15, device='cuda',
                   change_strategy=uniform_change_strategy):
    """
    Преобразует батч из DataLoader в формат, пригодный для подачи в модель.
    Случайным образом выбирает индексы для маски, изменения и сохранения, а затем применяет стратегию.
    """
    batch_size, cnt_words = batch['to_address'].shape
    msk_ind, change_ind, save_ind = get_n_random_splits(np.arange(cnt_words), [p_msk, p_change, p_save])
    volumes_features = batch['numeric_features']
    cnt_change = len(change_ind)
    volumes_features[:, change_ind] = torch.unsqueeze(change_strategy((batch_size, cnt_change)), 2)
    
    return msk_ind, change_ind, save_ind, dict(
        numeric_features=volumes_features.to(device),
        from_address=batch['from_address'].to(device),
        to_address=batch['to_address'].to(device),
        time_features=batch['time_features'].to(device),
        msk_ind=msk_ind + 1,
        volumes=volumes_features.to(device),
    )

# =============================================================================
# 5. Фабрики (Factories)
# =============================================================================

class PredictorFactory:
    """Фабрика для создания функции предиктора по названию."""
    @staticmethod
    def get_predictor(name: str = 'scalar_dot_predictor'):
        if name == 'scalar_dot_predictor':
            return scalar_dot_predictor
        elif name == 'time_cross_predictior':
            return time_cross_predictor
        else:
            raise ValueError(f"Неизвестный тип предиктора: {name}")

class LossFactory:
    """Фабрика для создания функции потерь по названию."""
    @staticmethod
    def get_loss_function(name: str = 'criterion'):
        if name == 'criterion':
            return criterion_loss_fn
        elif name == 'result_loss_slower_change':
            return result_loss_slower_change
        elif name == 'empty':
            return result_loss_empty
        else:
            raise ValueError(f"Неизвестный тип функции потерь: {name}")

class ChangeStrategyFactory:
    """Фабрика для создания стратегии изменения данных по названию."""
    @staticmethod
    def get_strategy(name: str = 'uniform'):
        if name == 'uniform':
            return uniform_change_strategy
        else:
            raise ValueError(f"Неизвестная стратегия изменения: {name}")

# =============================================================================
# 6. Класс конфигурации (Config) с проверкой параметров
# =============================================================================

class Config:
    """
    Класс-конфигуратор для обучения.
    
    При инициализации задаются настройки с дефолтными значениями, затем
    они объединяются с переданными параметрами и проверяются.
    Строковые ключи для выбора функций преобразуются в реальные функции с помощью фабрик.
    """
    def __init__(self, **config):
        self.config = config
        self.validate()
        # Преобразуем строковые ключи в реальные функции
        self.config['change_strategy'] = ChangeStrategyFactory.get_strategy(self.config['change_strategy'])
        self.config['loss_fn'] = LossFactory.get_loss_function(self.config['loss_fn'])
        self.config['result_loss'] = LossFactory.get_loss_function(self.config['result_loss'])
        self.config['predictor'] = PredictorFactory.get_predictor(self.config['model_predictor'])

    def validate(self):
        if self.config.get('num_epochs', 0) <= 0:
            raise ValueError("num_epochs должен быть положительным числом")
        for key in ['p_change', 'p_msk', 'p_save']:
            if not (0 <= self.config.get(key, 0) <= 1):
                raise ValueError(f"{key} должен быть в диапазоне [0, 1]")

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value
        self.validate()

    def as_dict(self):
        return self.config.copy()

# =============================================================================
# 7. Класс Trainer
# =============================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, run, config: Config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run = run
        self.device = device
        self.config = config.as_dict()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = scheduler_creator(
            self.optimizer,
            warmup_epochs=self.config['warmup_epochs'],
            gamma=self.config['gamma'],
            step_size=self.config['step_size']
        )

        self.all_train_losses = []
        self.all_val_losses = []
        self.all_train_loss_cross_time = []
        self.all_val_loss_cross_time = []
        self.last_time = pd.Timestamp.now()

    # ========= Общие утилиты =========

    def get_use_compositor(self):
        return (
            self.model.module.use_compositor
            if isinstance(self.model, nn.DataParallel)
            else self.model.use_compositor
        )

    def forward_and_predict(self, model_input):
        res = self.model(**model_input)
        volumes_pred = self.config['predictor'](
            cls_result=res['cls_result'],
            result=res['result'],
            from_emb=res['from_emb'],
            to_emb=res['to_emb'],
            use_compositor=self.get_use_compositor(),
            model=self.model,
        )
        return res, volumes_pred

    def process_batch(self, batch):
        return batch_to_model(
            batch=batch,
            p_msk=self.config['p_msk'],
            p_change=self.config['p_change'],
            p_save=self.config['p_save'],
            device=self.device,
            change_strategy=self.config['change_strategy'],
        )

    def compute_losses(self, volumes_pred, target, msk_ind, change_ind, save_ind, result):
        loss_pred = self.config['loss_fn'](
            pred=volumes_pred,
            target=target,
            msk_ind=msk_ind,
            change_ind=change_ind,
            criterion=nn.MSELoss(),
            save_ind=save_ind,
        )
        loss_time = self.config['result_loss'](
            result=result,
            coef=self.config['time_coef_loss'],
        ).to(volumes_pred.device)
        return loss_pred, loss_time

    # ========= Основные этапы =========

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_loss_cross_time = 0.0

        for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}"):
            self.optimizer.zero_grad()

            msk_ind, change_ind, save_ind, model_input = self.process_batch(batch)
            res, volumes_pred = self.forward_and_predict(model_input)
            loss_pred, loss_time = self.compute_losses(
                volumes_pred=volumes_pred,
                target=batch['value'].to(self.device),
                msk_ind=msk_ind,
                change_ind=change_ind,
                save_ind=save_ind,
                result=res['result'],
            )

            loss = loss_pred + loss_time
            loss.backward()
            self.optimizer.step()

            train_loss += loss_pred.detach().cpu().item()
            train_loss_cross_time += loss_time.detach().cpu().item()

        return train_loss, train_loss_cross_time

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_loss_cross_time = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validate Epoch {epoch + 1}"):
                msk_ind, change_ind, save_ind, model_input = self.process_batch(batch)
                res, volumes_pred = self.forward_and_predict(model_input)
                loss_pred, loss_time = self.compute_losses(
                    volumes_pred=volumes_pred,
                    target=batch['value'].to(self.device),
                    msk_ind=msk_ind,
                    change_ind=change_ind,
                    save_ind=save_ind,
                    result=res['result'],
                )

                val_loss += loss_pred.detach().cpu().item()
                val_loss_cross_time += loss_time.detach().cpu().item()

        return val_loss, val_loss_cross_time

    def log_metrics(self, epoch, train_loss, train_time, val_loss, val_time):
        avg_train = train_loss / len(self.train_loader)
        avg_val = val_loss / len(self.val_loader)
        avg_train_time = train_time / len(self.train_loader)
        avg_val_time = val_time / len(self.val_loader)

        self.all_train_losses.append(avg_train)
        self.all_val_losses.append(avg_val)
        self.all_train_loss_cross_time.append(avg_train_time)
        self.all_val_loss_cross_time.append(avg_val_time)

        self.scheduler.step(avg_val + avg_val_time)

        log_dict = {
            'epoch': epoch + 1,
            'train_loss': avg_train,
            'val_loss': avg_val,
            'train_loss_cross_time': avg_train_time,
            'val_loss_cross_time': avg_val_time,
            'lr': self.scheduler.get_last_lr(),
        }

        self.run.log(log_dict)
        print(f"Epoch [{epoch + 1}/{self.config['num_epochs']}]: "
              f"Train: {avg_train:.4f}, Val: {avg_val:.4f}, "
              f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

    # ========= Визуализация =========

    def plot_results(self, epoch):
        graphs_names = [
            "Все результаты",
            f"Последние {self.config['cnt_last_for_show']} train loss",
            f"Последние {self.config['cnt_last_for_show']} time loss",
            "Gramm matrix of known address embedings",
            "Predicts vs target (val)"
        ]
        lines_names = ['Train loss pred', 'Validate loss pred', 'Train loss time', 'Validate loss time']
        figs = {name: go.Figure() for name in graphs_names}

        fig = make_subplots(rows=5, cols=1, subplot_titles=graphs_names)

        # --- Loss curves ---
        for trace in [
            go.Scatter(x=np.arange(epoch + 1), y=self.all_train_losses, name=lines_names[0], mode='lines'),
            go.Scatter(x=np.arange(epoch + 1), y=self.all_val_losses, name=lines_names[1], mode='lines'),
            go.Scatter(x=np.arange(epoch + 1), y=self.all_train_loss_cross_time, name=lines_names[2], mode='lines'),
            go.Scatter(x=np.arange(epoch + 1), y=self.all_val_loss_cross_time, name=lines_names[3], mode='lines'),
        ]:
            fig.add_trace(trace, row=1, col=1)
            figs[graphs_names[0]].add_trace(trace)

        # --- Last N losses ---
        N = self.config['cnt_last_for_show']
        for trace in [
            go.Scatter(y=self.all_train_losses[-N:], name=lines_names[0], mode='lines'),
            go.Scatter(y=self.all_val_losses[-N:], name=lines_names[1], mode='lines'),
        ]:
            fig.add_trace(trace, row=2, col=1)
            figs[graphs_names[1]].add_trace(trace)

        for trace in [
            go.Scatter(y=self.all_train_loss_cross_time[-N:], name=lines_names[2], mode='lines'),
            go.Scatter(y=self.all_val_loss_cross_time[-N:], name=lines_names[3], mode='lines'),
        ]:
            fig.add_trace(trace, row=3, col=1)
            figs[graphs_names[2]].add_trace(trace)

        # --- Address emb heatmap ---
        with torch.no_grad():
            emb = self.model.address_embedding.weight.detach().cpu()
        heat = go.Heatmap(z=emb @ emb.T, colorbar=dict(len=0.2, y=0.3, yanchor='middle', x=1.05))
        fig.add_trace(heat, row=4, col=1)
        figs[graphs_names[3]].add_trace(go.Heatmap(z=emb @ emb.T))

        # --- Predict vs target ---
        with torch.no_grad():
            it = iter(self.train_loader)
            raw_show_batch = []
            it = iter(self.train_loader)

            raw_show_batch.append(next(it))
            batch_size = raw_show_batch[0]['time_features'].shape[0]
            num_batches = self.config['show_batch_size'] // batch_size

            for _ in range(num_batches):
                raw_show_batch.append(next(it))

            keys = raw_show_batch[0].keys()
            show_batch = {k: torch.cat([b[k] for b in raw_show_batch], dim=0) for k in keys}

            msk_ind, change_ind, save_ind, model_input = self.process_batch(show_batch)
            res, volumes_pred = self.forward_and_predict(model_input)

            targets = show_batch['numeric_features'][..., -1].detach().cpu().reshape(-1)
            preds = volumes_pred.reshape(-1).detach().cpu()
            indexes = show_batch['time_features'][..., -1].detach().cpu().reshape(-1)
            idx_sorted = sorted(range(len(indexes)), key=lambda x: indexes[x])[:self.config['show_batch_size']]

            fig.add_trace(go.Scatter(x=np.arange(len(idx_sorted)), y=targets[idx_sorted], name="Targets", line_shape="hv"), row=5, col=1)
            fig.add_trace(go.Scatter(x=np.arange(len(idx_sorted)), y=preds[idx_sorted], name="Predicts", line_shape="hv"), row=5, col=1)

            figs[graphs_names[4]].add_trace(go.Scatter(y=targets[idx_sorted], name="Targets"))
            figs[graphs_names[4]].add_trace(go.Scatter(y=preds[idx_sorted], name="Predicts"))

        fig.update_layout(width=1000, height=1500)
        if epoch % self.config['loggin_each'] == 0:
            self.run.log(figs)

        if self.config['show_img']:
            clear_output(wait=True)
            fig.show()

    # ========= Главный цикл =========

    def train(self):
        for epoch in range(self.config['start_epoch'], self.config['num_epochs']):
            np.random.seed(None)
            train_loss, train_time = self.train_epoch(epoch)
            np.random.seed(42)
            val_loss, val_time = self.validate_epoch(epoch)

            self.log_metrics(epoch, train_loss, train_time, val_loss, val_time)

            time_to_show = pd.Timestamp.now() - self.last_time > pd.Timedelta(seconds=self.config['seconds_betwen_image_show'])
            if time_to_show or epoch % self.config['loggin_each'] == 0:
                self.last_time = pd.Timestamp.now()
                self.plot_results(epoch)
