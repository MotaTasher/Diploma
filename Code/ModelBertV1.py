import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from IPython.display import clear_output


class BertForTransactionRegressionV1(nn.Module):
    def __init__(self, known_address_len, emb_size=64, use_compositor=False, num_attention_heads=12,
                 time_features=1):
        super(BertForTransactionRegressionV1, self).__init__()

        config = BertConfig(num_attention_heads=num_attention_heads)
        self.use_compositor = use_compositor
        
        hidden_size = (time_features + emb_size + 1) * 2
        config.hidden_size = hidden_size
        self.bert = BertModel(config)
        self.address_embedding = nn.Embedding(num_embeddings=known_address_len + 1, embedding_dim=emb_size)

        cls_emb = torch.randn((hidden_size))
        # msk_emb = torch.randn((hidden_size))
        self.cls_token = nn.Parameter(cls_emb)
        # self.msk_token = nn.Parameter(msk_emb)

        self.linear = nn.Linear(hidden_size, emb_size if self.use_compositor else emb_size * 2)
        if self.use_compositor:
            self.compositor = nn.Parameter(torch.Tensor(emb_size + 1, emb_size + 1, emb_size + 1))
        
        self.feature_matrix = nn.Parameter(torch.Tensor(1, time_features))
            
    def get_time_features_emb(self, time_features):
        features = time_features @ self.feature_matrix
        res = torch.concat([torch.cos(features), torch.sin(features)], dim=-1)
        return res

    def forward(self, numeric_features, from_address, to_address, time_features, msk_ind, volumes):
        from_emb = self.address_embedding(from_address)
        to_emb = self.address_embedding(to_address)
        
        time_features = self.get_time_features_emb(time_features)
        features = torch.cat([from_emb, to_emb, time_features, volumes, torch.ones_like(volumes)], dim=-1)
        batch_size, cnt_words, data_shape = features.shape

        cls_data = torch.tile(self.cls_token, (batch_size, 1, 1))
        features = torch.concat([cls_data, features], axis=1)

        # может маскировать надо только объемы а не всё сразу скрывая от модели вообще всё ????
        # ещё и разные индексы брать, но это уже не обязательно
        features[:, msk_ind, -1] = 0
        features[:, msk_ind, -2] = 0

        bert_output = self.bert(inputs_embeds=features).last_hidden_state
        result = self.linear(bert_output)

        cls_result = result[:, 0, :]
        result = result[:, 1:, :]

        return dict(cls_result=cls_result, result=result,
                    from_emb=from_emb, to_emb=to_emb)
