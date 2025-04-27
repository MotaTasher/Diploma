import pandas as pd
import numpy as np
import json
import typing as tp
from tqdm.auto import tqdm
import os
import Code.Logger as LogsLib
import torch.nn as nn
from functools import partial
import numba
import torch
import torch.nn.functional as F


ETH_RPC_URL="https://mainnet.infura.io/v3/e0e69a0f0dae4450889fa8fa17117760"
BLOCKS = "18M:+100"
ARGC = "echo $ETH_RPC_URL"

wei_to_eth = 1e18


@numba.njit
def CalculateCoef(boom_times, sigma, coef, time):
    total_coef = np.zeros_like(time)
    for boom_time in boom_times:
        total_coef += np.exp((-(time - boom_time) ** 2 / (2 * sigma))) * coef
    return total_coef

class TimeBoomStrategyV1():
    def __init__(self, boom_count, min_time, max_time, sigma=1,
                 coef=1, batch_size=100, device='cuda'):
        self.device = device
        self.boom_times = torch.tensor(np.random.uniform(min_time, max_time, boom_count)).to(self.device)
        self.sigma = sigma
        self.coef = coef
        self.batch_size = batch_size

    def __call__(self, time):
        if len(self.boom_times) == 0:
            return np.ones_like(time)
        else:
            total_coef = torch.zeros(time.shape).to(self.device)
            # for boom_time in tqdm(self.boom_times, total=len(self.boom_times)):
            #     total_coef += np.exp((-(time - boom_time) ** 2 / (2 * self.sigma))) * self.coef
            batch_size = self.batch_size
            time = torch.tensor(time).reshape(1, -1).to(self.device)
            for i in tqdm(range(0, len(self.boom_times), batch_size)):
                bt_batch = self.boom_times[i:i + batch_size].reshape(-1, 1)
                time_diff = time - bt_batch
                batch_result = torch.exp(-time_diff ** 2 / (2 * self.sigma))
                total_coef += torch.sum(batch_result, axis=0) * self.coef
            return total_coef.detach().cpu().numpy()


class AuthorStrategyV1():
    def __init__(self, dim, cnt, generate_strategy=np.random.uniform, params=dict(low=0, high=1)):
        self.people_count = cnt
        self.dimension = dim
        self.generate_strategy = partial(generate_strategy, **params)
        self.people_emb = self.generate_strategy(size=(cnt, dim))

    def __call__(self, indexes):
        result = self.generate_strategy(size=(len(indexes), self.dimension))
        result[indexes < self.people_count] = np.take(self.people_emb, indexes[indexes < self.people_count], axis=0)
        return result


def SimpleDataloader(cnt, logger_name='logs/raw_logger.log', logger_fake=None, seed=42, count_people=2,
                     volume_strategy=np.random.randint, volume_strategy_param=dict(low=0, high=2),
                     time_strategy=np.random.uniform, time_strategy_param=dict(low=1, high=3)):
    with LogsLib.LoggerCreate(logger_fake) as logger:
        logger.Write(f'Use seed {seed}')
        np.random.seed(seed)
        df = pd.DataFrame(index=range(cnt), columns=['from', 'to', 'value', 'timestamp'])
        df.loc[:, 'from'] = np.random.randint(0, count_people, cnt)
        df.loc[:, 'to'] = np.random.randint(0, count_people - 1, cnt)
        df.loc[df.loc[:, 'to'] >= df.loc[:, 'from'], 'to'] += 1
        df.loc[:, 'value'] = volume_strategy(**volume_strategy_param, size=cnt)
        df.loc[:, 'timestamp'] = np.cumsum(time_strategy(**time_strategy_param, size=cnt))
        return df

    
def DatasetTimeBoomAuthorDependence(cnt, logger_name='logs/raw_logger.log', logger_fake=None, seed=42,
                                    count_people=2, boom_strategy_fabric=TimeBoomStrategyV1, cnt_boom=10, sigma=1,
                                    author_strategy_fabric=AuthorStrategyV1,
                                    time_strategy=np.random.uniform, time_strategy_param=dict(low=1, high=3), time_mean=2):
    with LogsLib.LoggerCreate(logger_fake) as logger:
        logger.Write(f'Use seed {seed}')
    
        np.random.seed(seed=seed)
        boom_strategy = boom_strategy_fabric(cnt_boom, 0, time_mean * cnt, sigma=sigma)
        author_strategy = author_strategy_fabric(16, count_people)
        df = pd.DataFrame(index=range(cnt), columns=['from', 'to', 'value', 'timestamp'])
        df.loc[:, 'from'] = np.random.randint(0, count_people, cnt)
        df.loc[:, 'to'] = np.random.randint(0, count_people - 1, cnt)
        df.loc[df.loc[:, 'to'] >= df.loc[:, 'from'], 'to'] += 1
        df.loc[:, 'timestamp'] = np.cumsum(time_strategy(**time_strategy_param, size=cnt))
        df.loc[:, 'value'] = np.sum(author_strategy(df.loc[:, 'from']) * author_strategy(df.loc[:, 'to']), axis=1)
        df.loc[:, 'from'] = df.loc[:, 'from'].astype(int)
        df.loc[:, 'to'] = df.loc[:, 'to'].astype(int)
        df.loc[:, 'value'] = df.loc[:, 'value'].astype(float)
        df.loc[:, 'value'] *= boom_strategy(df.loc[:, 'timestamp'].values.astype(float))
        df.loc[:, 'timestamp'] = df.loc[:, 'timestamp'].astype(float)
        return df

def DatasetLimitMoneyVolume(cnt, logger_name='logs/raw_logger.log', logger_fake=None, seed=42,
                            count_people=2, people_lambda: float | tp.Iterable[float]=1,
                            time_strategy=np.random.uniform, time_strategy_param=dict(low=1, high=3)):
    with LogsLib.LoggerCreate(logger_fake) as logger:
        logger.Write(f'Use seed: {seed}')
        logger.Write(f'Use DatasetLimitMoneyVolume')
        np.random.seed(seed=seed)

        df = pd.DataFrame(index=range(cnt), columns=['from', 'to', 'value', 'timestamp'])
        df.loc[:, 'timestamp'] = np.cumsum(time_strategy(**time_strategy_param, size=cnt))
        
        balances = torch.zeros(count_people, requires_grad=False)
        
        data = df.values
        
        second_men = np.random.randint(0, count_people - 1, cnt)
        prev_ts = 0
        for ind in range(cnt):
            ts = data[ind, -1]
            balances += np.random.poisson(people_lambda * (ts - prev_ts), count_people)
            prev_ts = ts
            probs = F.softmax(balances, dim=0).numpy() * (balances.numpy() != 0)
            if probs.sum() == 0:
                probs[0] += 1

            probs /= probs.sum()
            first_man = np.random.choice(range(0, count_people), p= probs)
            second_man = second_men[ind]
            
            spent = balances[first_man] * np.random.uniform(0, 1)
            balances[first_man] -= spent
            balances[second_man] += spent
            if second_man >= first_man:
                second_man += 1
            data[ind] = np.array([first_man, second_man, spent, ts])
            
    return pd.DataFrame(data, index=range(cnt), columns=['from', 'to', 'value', 'timestamp'])
            

def GetJsonByInd(ind, path='data', logger_name='logs/raw_logger.log', logger_fake=None):
    with LogsLib.LoggerCreate(logger_fake) as logger:
        real_path = os.path.join(path, f'dump_{ind}.json')
        if not os.path.exists(real_path):
            logger.Write(f"Write json for {ind}, in path: '{real_path}'")
            logger.Write("Start download")
            os.system(f"curl https://mainnet.infura.io/v3/e0e69a0f0dae4450889fa8fa17117760 -X POST -H 'Content-Type: application/json' "
                    f"-d '{{\"jsonrpc\":\"2.0\", \"method\":\"eth_getBlockByNumber\", \"params\":[\"0x{ind:X}\", true], \"id\":1}}' > {real_path} 2>> {logger_name}")
            logger.Write("Downloaded")
        if not os.path.exists(real_path):
            logger.Write(f"Some problems with downloading, return None (in path {real_path})")    
            return None
        json_line = ''.join(open(real_path, 'r').readlines())
        if len(json_line) == 0:
            return None
        try:
            json_data = json.loads(json_line)
            return json_data
        except json.JSONDecodeError as exc:
            logger.Write(f"Has exception {exc.args} while loads json")
        except TypeError as exc:
            logger.Write(f"Has exception {exc.args} while loads json")
        except Exception as exc:
            raise exc
            

useful_columns = ['timestamp', 'from', 'to', 'value']

def GetTransactionsByInd(ind: int | tp.Iterable[int], path='data', logger_fake=None, convert_to_df=True) -> pd.DataFrame | None:
    with LogsLib.LoggerCreate(logger_fake) as logger:
        if type(ind) is int:
            res = GetJsonByInd(int(ind), path, logger_fake=logger)
            if res is not None:
                if len(res['result']['transactions']) == 0:
                    return None
                if convert_to_df:
                    df = pd.DataFrame(res['result']['transactions'])
                    df['timestamp'] = pd.Timestamp(int(res['result']['timestamp'], 16) * 1e9)
                else:
                    return pd.Timestamp(int(res['result']['timestamp'], 16) * 1e9), res['result']['transactions']
            else:
                return None
        else:
            data = []
            ts = []
            for real_ind in tqdm(ind):
                res = GetTransactionsByInd(real_ind, path, logger_fake=logger, convert_to_df=False)
                if res is not None:
                    next_ts, next_df = res
                    if next_df is not None:
                        data += next_df
                        ts += [next_ts for _ in range(len(next_df))]
            df = pd.DataFrame(data)

            ind = 0
            df['timestamp'] = ts
            if len(df) == 0:
                return None

        if convert_to_df:
            convert_hex_to_int = lambda x: int(x, 16)
            for columns_name in useful_columns:
                if columns_name == 'timestamp':
                    continue
                column_value = df.loc[:, columns_name]
                df.loc[~column_value.isna(), columns_name] = df.loc[~column_value.isna(), columns_name].apply(convert_hex_to_int)
            df.value = df.value.astype(float) / wei_to_eth

        return df.reset_index(drop=True)