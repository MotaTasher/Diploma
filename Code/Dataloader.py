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
import pyarrow.lib

import os
import json
import time
import requests
import pandas as pd
# from tqdm import tqdm
import Code.Logger as LogsLib

import aiohttp
import asyncio
import aiofiles
import pickle

WEI_TO_ETH = 1e18

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
                     time_strategy=np.random.uniform, time_strategy_param=dict(low=1, high=3),
                     address_limit=None):
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
                                    time_strategy=np.random.uniform, time_strategy_param=dict(low=1, high=3), time_mean=2,
                                    address_limit=None):
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
                            time_strategy=np.random.uniform, time_strategy_param=dict(low=1, high=3),
                            address_limit=None):
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
            os.system(f"curl {ETH_RPC_URL} -X POST -H 'Content-Type: application/json' "
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


class SyncBatchEthFetcher:
    def __init__(
        self,
        rpc_url: str = ETH_RPC_URL,
        batch_size: int = 100,
        logger_name: str = "logs/raw_logger.log",
        logger_fake=None,
        cache_dir: str = "block_cache"
    ):
        self.rpc_url = rpc_url
        self.batch_size = batch_size
        self.logger_name = logger_name
        self.logger_fake = logger_fake
        self.cache_dir = cache_dir
        self.total_df_folder = os.path.join(self.cache_dir, 'total_df')
        if not os.path.exists(self.total_df_folder):
            os.makedirs(self.total_df_folder, exist_ok=True)

        os.makedirs(cache_dir, exist_ok=True)

    def _batch_cache_path(self, batch_start: int, batch_end: int) -> str:
        return os.path.join(self.cache_dir, f"blocks_{batch_start}_{batch_end}.json")

    def _is_negative_response(self, response_json: list) -> bool:
        return any(isinstance(r, dict) and r.get("error") for r in response_json)

    def fetch_blocks(self, block_nums: list[int]) -> list[dict]:
        all_results = []

        with LogsLib.LoggerCreate(self.logger_fake) as logger:
            logger.Write(f"Fetching {len(block_nums)} blocks in batches of {self.batch_size}")

            for i in tqdm(range(0, len(block_nums), self.batch_size)):
                batch = block_nums[i : i + self.batch_size]
                batch_start = batch[0]
                batch_end = batch[-1]
                cache_file = self._batch_cache_path(batch_start, batch_end)

                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                            logger.Write(f"Loaded cached batch {batch_start}-{batch_end}")
                            all_results.extend(cached_data)
                            continue
                    except json.JSONDecodeError as e:
                        logger.Write(f"Corrupted cache file {cache_file}, removing it. Error: {e}")
                        os.remove(cache_file)

                payload = [
                    {
                        "jsonrpc": "2.0",
                        "id": idx,
                        "method": "eth_getBlockByNumber",
                        "params": [hex(n), True]
                    }
                    for idx, n in enumerate(batch)
                ]

                try:
                    response = requests.post(self.rpc_url, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    if self._is_negative_response(data):
                        logger.Write(f"Negative response in batch {batch_start}-{batch_end}, stopping.")
                        break  # Остановить весь процесс

                    # Save to cache
                    with open(cache_file, 'w') as f:
                        json.dump(data, f)

                    logger.Write(f"Fetched and cached batch {batch_start}-{batch_end}")
                    all_results.extend(data)

                    time.sleep(0.1)  # немного замедлить

                except Exception as e:
                    logger.Write(f"Error fetching batch {batch_start}-{batch_end}: {e}")
                    continue

        return all_results

    def _get_cached_batches(self) -> set[str]:
        return set(os.listdir(self.cache_dir))

    
    async def fetch_blocks_async(self, block_nums: list[int]) -> list[dict]:
        all_results = []
        async def _load_all_cached(self, cached_files: set[str]) -> dict[str, list[dict]]:
            from aiofiles import open as aio_open

            async def load_file(filename):
                full_path = os.path.join(self.cache_dir, filename)
                async with aio_open(full_path, 'r') as f:
                    content = await f.read()
                    return filename, json.loads(content)

            tasks = [load_file(fname) for fname in cached_files]
            results = await asyncio.gather(*tasks)

            return dict(results)  # filename → list of blocks


        async with aiohttp.ClientSession() as session:
            cached_files = self._get_cached_batches()

            for i in tqdm(range(0, len(block_nums), self.batch_size), desc="Async Fetching"):
                batch = block_nums[i : i + self.batch_size]
                batch_start, batch_end = batch[0], batch[-1]

                cache_file_name = f"blocks_{batch_start}_{batch_end}.json"
                cache_file_path = os.path.join(self.cache_dir, cache_file_name)

                if cache_file_name in cached_files:
                    # with open(cache_file_path, 'r') as f:
                    #     cached_data = json.load(f)
                    #     all_results.extend(cached_data)
                    #     continue
                    # async with aiofiles.open(cache_file_path, 'r') as f:

                    with open(cache_file_path, "rb") as f:
                        data = pickle.load(f)
                        json_data = await f.read()
                        cached_data = json.loads(json_data)
                        all_results.extend(cached_data)
                        continue

                payload = [
                    {
                        "jsonrpc": "2.0",
                        "id": idx,
                        "method": "eth_getBlockByNumber",
                        "params": [hex(n), True]
                    }
                    for idx, n in enumerate(batch)
                ]

                try:
                    async with session.post(self.rpc_url, json=payload) as resp:
                        data = await resp.json()
                        if self._is_negative_response(data):
                            break

                        # with open(cache_file_path, 'w') as f:
                        #     json.dump(data, f)
                        with open(cache_file_path, "wb") as f:
                            pickle.dump(data, f)

                        all_results.extend(data)
                        await asyncio.sleep(0.3)  # throttle

                except Exception as e:
                    print(f"Error in async batch {batch_start}-{batch_end}: {e}")
                    continue

        return all_results

    @staticmethod
    def blocks_json_to_df(blocks_json: list[dict]) -> pd.DataFrame:
        records = []
        for entry in blocks_json:
            blk = entry.get("result")
            if not blk or "transactions" not in blk:
                continue
            ts = int(blk["timestamp"], 16)
            for tx in blk["transactions"]:
                records.append({
                    "from": int(tx["from"], 16),
                    "to": int(tx["to"], 16) if tx.get("to") else None,
                    "value": int(tx["value"], 16) / WEI_TO_ETH,
                    "timestamp": ts
                })
        return pd.DataFrame.from_records(records, columns=["from", "to", "value", "timestamp"])
    
    
    def get_transactions(self, start_block: int, end_block: int, use_async=False, save_df=True, use_cache=True) -> pd.DataFrame:
        dataset_cache_path = os.path.join(self.total_df_folder, f"dataset_{start_block}_{end_block}.feather")

        if use_cache and os.path.exists(dataset_cache_path):
            try:

                df = pd.read_feather(dataset_cache_path)
                df["from"] = df["from"].apply(lambda x: int(x, 16))
                df["to"] = df["to"].apply(lambda x: int(x, 16) if pd.notnull(x) else None)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                return df
            except pyarrow.lib.ArrowInvalid as e:
                print(f"[WARNING] Failed to read cached dataset: {e}. Recomputing...")
                os.remove(dataset_cache_path)

        block_list = list(range(start_block, end_block))
        if use_async:
            blocks_json = asyncio.run(self.fetch_blocks_async(block_list))
        else:
            blocks_json = self.fetch_blocks(block_list)

        df = self.blocks_json_to_df(blocks_json)
        df.timestamp = pd.to_datetime(df.timestamp, unit="s")

        if save_df:
            df_copy = df.copy()
            df_copy["from"] = df_copy["from"].apply(lambda x: hex(x))
            df_copy["to"] = df_copy["to"].apply(lambda x: hex(x) if pd.notnull(x) else None)
            df_copy.reset_index(drop=True).to_feather(dataset_cache_path)
        return df.sort_values("timestamp").reset_index(drop=True)

        


def GetEthereumDataset(
    start_block: int,
    end_block: int,
    address_limit: int = None,
    logger_fake=None,
    logger_name="logs/eth_fetch.log",
    cache_dir="block_cache",
    batch_size=500,
    cnt=None,
    use_async=False,
) -> pd.DataFrame:

    fetcher = SyncBatchEthFetcher(
        rpc_url=ETH_RPC_URL,
        batch_size=batch_size,
        logger_name=logger_name,
        logger_fake=logger_fake,
        cache_dir=cache_dir,
    )

    df = fetcher.get_transactions(start_block, end_block, use_async=use_async)
    df = df.dropna(subset=["to"])
    df = df[df["value"] > 0]

    volume_by_address = pd.concat([
        df[["from", "value"]].rename(columns={"from": "address"}),
        df[["to", "value"]].rename(columns={"to": "address"})
    ])

    volume_sum = volume_by_address.groupby("address")["value"].sum()
    top_addresses = volume_sum.sort_values(ascending=False).head(address_limit).index.tolist()

    address_map = {addr: idx for idx, addr in enumerate(top_addresses)}
    default_index = address_limit

    df["from"] = df["from"].apply(lambda x: address_map.get(x, default_index))
    df["to"]   = df["to"].apply(lambda x: address_map.get(x, default_index))

    df["timestamp"] = (df["timestamp"] - pd.Timestamp(0)).dt.total_seconds().astype(float)

    df = df.astype({"from": int, "to": int, "value": float, "timestamp": float})
    return df
    df = df.sort_values("timestamp").reset_index(drop=True)
