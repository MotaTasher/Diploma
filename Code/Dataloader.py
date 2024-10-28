import pandas as pd
import numpy as np
import json
import typing as tp
from tqdm.notebook import tqdm
import os
import Code.Logger as logs


ETH_RPC_URL="https://mainnet.infura.io/v3/e0e69a0f0dae4450889fa8fa17117760"
BLOCKS = "18M:+100"
ARGC = "echo $ETH_RPC_URL"

wei_to_eth = 1e18


def GetJsonByInd(ind, path='data', logger_name = 'logs/raw_logger.log', logger_fake = None):
    with logs.LoggerCreate(logger_fake) as logger:
        real_path = os.path.join(path, f'dump_{ind}.json')
        if not os.path.exists(real_path):
            logger.Write(f"Write json for {ind}, in path: '{real_path}'")
            logger.Write("Start download")
            os.system(f"curl https://mainnet.infura.io/v3/e0e69a0f0dae4450889fa8fa17117760 -X POST -H 'Content-Type: application/json' "
                    f"-d '{{\"jsonrpc\":\"2.0\", \"method\":\"eth_getBlockByNumber\", \"params\":[\"0x{ind:X}\", true], \"id\":1}}' > {real_path} 2>> {logger_name}")
            logger.Write(logger,"Downloaded")
        if not os.path.exists(real_path):
            logger.Write(f"Some problems with downloading, return None (in path {real_path})")    
            return None
        json_line = ''.join(open(real_path, 'r').readlines())
        if len(json_line) == 0:
            return None
        try:
            json_data = json.loads(json_line)
            return json_data
        except Exception as exc:
            logger.Write(f"Has exception {exc.args} while loads json")
            

useful_columns = ['timestamp', 'from', 'to', 'nonce', 'transactionIndex', 'type', 'value']

def GetTransactionsByInd(ind: int | tp.Iterable[int], path='data', logger_fake=None) -> pd.DataFrame | None:
    with logs.LoggerCreate(logger_fake) as logger:
        if type(ind) is int:
            res = GetJsonByInd(int(ind), path, logger_fake=logger)
            if res is not None:
                if len(res['result']['transactions']) == 0:
                    return None
                df = pd.DataFrame(res['result']['transactions'])
                df['timestamp'] = pd.Timestamp(int(res['result']['timestamp'], 16) * 1e9)
                convert_hex_to_int = lambda x: int(x, 16)
                for columns_name in useful_columns:
                    if columns_name == 'timestamp':
                        continue
                    column_value = df.loc[:, columns_name]
                    df.loc[~column_value.isna(), columns_name] = df.loc[~column_value.isna(), columns_name].apply(convert_hex_to_int)
                df.value = df.value.astype(float) / wei_to_eth
            else:
                return None
        else:
            res_dfs = []
            for real_ind in tqdm(ind):
                next_df = GetTransactionsByInd(real_ind, path, logger_fake=logger)
                if next_df is not None:
                    res_dfs.append(next_df)
            df = pd.concat(res_dfs).reset_index(drop=True)
            if len(df) == 0:
                return None

        return df