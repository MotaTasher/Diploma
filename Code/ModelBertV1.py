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


class BertForTransactionRegression(nn.Module):
    def __init__(self, known_address_len, emb_size=64):
        super(BertForTransactionRegression, self).__init__()

        config = BertConfig()
        
        hidden_size =  emb_size * 2 + 4
        config.hidden_size = hidden_size
        self.bert = BertModel(config)
        self.address_embedding = nn.Embedding(num_embeddings=known_address_len + 1, embedding_dim=emb_size)

        cls_emb = torch.randn((hidden_size))
        msk_emb = torch.randn((hidden_size))
        self.cls_token = nn.Parameter(cls_emb)
        self.msk_token = nn.Parameter(msk_emb)

        self.linear = nn.Linear(hidden_size, emb_size * 2)
    
    def forward(self, numeric_features, from_address, to_address, time_features, msk_ind):
        from_emb = self.address_embedding(from_address)
        to_emb = self.address_embedding(to_address)
        
        features = torch.cat([numeric_features, from_emb, to_emb, time_features], dim=-1)
        batch_size, cnt_words, data_shape = features.shape

        cls_data = torch.tile(self.cls_token, (batch_size, 1, 1))

        features = torch.concat([cls_data, features], axis=1)

        features[:, msk_ind, :] = self.msk_token

        bert_output = self.bert(inputs_embeds=features).last_hidden_state
        result = self.linear(bert_output)

        cls_result = result[:, 0, :]
        result = result[:, 1:, :]

        return dict(cls_result=cls_result, result=result,
                    from_emb=from_emb, to_emb=to_emb)


