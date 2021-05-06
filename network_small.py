import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LipReadinNN_LSTM(nn.Module):
    def __init__(self, tokens_to_id):
        super(self.__class__, self).__init__()
        
        self.loss = None
        self.tokens_to_id = tokens_to_id
        self.id_to_tokens = {i: tok for tok, i in self.tokens_to_id.items()}
        self.embedding = nn.Embedding(num_embeddings=len(tokens_to_id), embedding_dim=512, padding_idx=tokens_to_id['_PAD_'])
        
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Identity()
        resnet18.requires_grad_(False)
        self.resnet18 = resnet18
        
        self.LSTM1 = nn.LSTM(input_size=512, hidden_size=1024, num_layers=2, dropout=0.1) # encoder on win_len
        self.LSTM_encoder = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2, dropout=0.1) # encoder on frames_num
        self.LSTM_decoder = nn.LSTM(input_size=512, hidden_size=512, num_layers=1) #decoder

        self.linear = nn.Linear(in_features=512, out_features=len(tokens_to_id))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input_batch, target_list_of_tokens=None):      
        batch_size, frames_num, win_len, color, height, width = input_batch.shape
        output = torch.moveaxis(input_batch, (0,1,2), (2,1,0)) # [win_len, frames_num, batch_size, color, height, width]
        
        output = output.reshape(win_len*frames_num*batch_size, color, height, width) # [batch, C, H, W] for resnet
        output = self.resnet18(output) # [batch, feature=512]
        
        output = output.reshape(win_len, frames_num*batch_size, 512) # [time, batch, feature=512] for LSTM1
        output, (h_n, c_n) = self.LSTM1(output) # output shape is [time, batch, feature=1024]
        output = output[-1, :, :] # last state output in LSTM seq [batch, feature=1024]
        
        output = output.reshape(frames_num, batch_size, 1024) # [frames_num, batch, feature=1024] for LSTM_encoder
        output, (h_n, c_n) = self.LSTM_encoder(output) # output shape is [frames_num, batch, feature=512]
        output = output[-1, :, :] # last state output in LSTM seq [batch, feature=512]
        
        dummy_h = output[None, :, :] # [1, batch, feature=512]
        dummy_z = torch.zeros_like(dummy_h) # [1, batch, feature=512]
        
        # если в nn передали list_of_tokens, то loss будет высчитываться, сравнивая с этим истинным значением
        # если в nn не передали list_of_tokens, то loss высчитываться не будет
        if target_list_of_tokens is not None:
            target_batch = self.list_of_tokens_to_tensor_of_tokens_idx(target_list_of_tokens) # [batch, seq_len]
            target_batch = target_batch.to(self.embedding.weight.device) # костыль, чтобы совпадало расположение тензоров на device

            target_output = self.embedding(target_batch) # [batch, seq_len, emb_size=512]
            target_output = torch.moveaxis(target_output, (0,1), (1,0)) # [seq_len, batch, emb_size=512] for LSTM_decoder
        else:
            dummy_batch = self.tokens_to_id['_BOS_']*torch.ones((batch_size, 1), dtype=torch.long) # [batch, seq_len=1]
            dummy_batch = dummy_batch.to(self.embedding.weight.device) # костыль, чтобы совпадало расположение тензоров на device

            target_output = self.embedding(dummy_batch) # [batch, seq_len=1, emb_size=512]
            target_output = torch.moveaxis(target_output, (0,1), (1,0)) # [seq_len=1, batch, emb_size=512] for LSTM_decoder
            
            # generate predict_output untill all seq hasn't _EOS_ or len(seq) != 30
            i = 1; max_iter = 30 #target_batch.shape[1] if target_list_of_tokens is not None else 30
            while i < max_iter:
                i += 1
                predict_output, (h_n, c_n) = self.LSTM_decoder(target_output, (dummy_h, dummy_z)) # [seq_len=i, batch, emb_size=512]
                # добавляем к target_output последний слайс из predict_output
                # до тех пор, пока не получим длину max_iter
                last_slice_of_predict_output = predict_output[-1:, :, :] # [seq_len=1, batch, emb_size=512]
                target_output = torch.cat((target_output, last_slice_of_predict_output), dim=0) # [seq_len=i+1, batch, emb_size=512]
                
                # проверка на наличие токена _EOS_ в ответе
                predict_output = torch.moveaxis(predict_output, (0,1), (1,0)) # [batch, seq_len=i, emb_size=512] for Linear
                predict_output = self.linear(predict_output) # [batch, seq_len=i, classes_num]
                probs = self.softmax(predict_output) # [batch, seq_len=i, classes_num]
                predict_tensor_of_tokens_idx = torch.argmax(probs, dim=-1) # [batch, seq_len=i]

                # сравниваем каждый токен в строке с токеном _EOS_ и берем логическое ИЛИ вдоль размерности seq
                val, ind = torch.max(predict_tensor_of_tokens_idx==self.tokens_to_id['_EOS_'], dim=1)
                # берем логическое И вдоль размерности batch, чтобы убедиться, что каждый пример из батча имеет токен _EOS_
                val, ind = torch.min(val, dim=0)

                # если в каждом примере есть токен _EOS_, то заканчиваем генерацию
                if val.data:
                    break
            # else:
                # # для фиктивного расчета loss
                # dummy_batch = torch.cat((dummy_batch, predict_tensor_of_tokens_idx), dim=1) # [batch, seq_len=max_iter]
                    
        # generate predict_output one time
        predict_output, (h_n, c_n) = self.LSTM_decoder(target_output, (dummy_h, dummy_z)) # [seq_len, batch, emb_size=512]
        predict_output = torch.moveaxis(predict_output, (0,1), (1,0)) # [batch, seq_len, emb_size=512] for Linear
        predict_output = self.linear(predict_output) # [batch, seq_len, classes_num]

        # for return
        probs = self.softmax(predict_output) # [batch, seq_len, classes_num]
        predict_tensor_of_tokens_idx = torch.argmax(probs, dim=-1) # [batch, seq_len]
        predict_list_of_tokens = self.tensor_of_tokens_idx_to_list_of_tokens(predict_tensor_of_tokens_idx)

        if target_list_of_tokens is not None:
            # for compute loss
            predict_batch = torch.moveaxis(predict_output, (1,2), (2,1)) # [batch, classes_num, seq_len] for loss
            self.loss = self.compute_loss(predict_batch[:,:,:-1], target_batch[:,1:])
        else:
            self.loss = None
            # # for compute dummy loss
            # predict_batch = torch.moveaxis(predict_output, (1,2), (2,1)) # [batch, classes_num, seq_len] for loss
            # self.loss = self.compute_loss(predict_batch[:,:,:-1], dummy_batch[:,:,:-1])
        
        return predict_list_of_tokens

    def list_of_tokens_to_tensor_of_tokens_idx(self, list_of_tokens): #list of sentences, sentense is list of tokens
        batch_num = len(list_of_tokens)
        max_sentence_len = max(len(sentence) for sentence in list_of_tokens) + 2 # for <BOS> and <EOS>
        
        tensor_of_tokens_idx = self.tokens_to_id['_PAD_']*torch.ones((batch_num, max_sentence_len), dtype=torch.long)
        for i, sentence in enumerate(list_of_tokens):
            tensor_of_tokens_idx[i, 0] = self.tokens_to_id['_BOS_']
            for j, token in enumerate(sentence):
                token_idx = self.tokens_to_id.get(token, self.tokens_to_id['_UNK_'])
                tensor_of_tokens_idx[i, j+1] = token_idx
            else:
                tensor_of_tokens_idx[i, j+2] = self.tokens_to_id['_EOS_']
        
        return tensor_of_tokens_idx
    
    def tensor_of_tokens_idx_to_list_of_tokens(self, tensor_of_tokens_idx): # tensor of tokens idx with shape of [batch, seq]
        list_of_tokens = []
        for i, sentence in enumerate(tensor_of_tokens_idx):
            tokens = []
            for j, idx in enumerate(sentence):
                tokens.append(self.id_to_tokens[idx.item()])
            list_of_tokens.append(tokens)
        
        return list_of_tokens
    
    def compute_loss(self, predict_batch, target_batch):
        loss_f = nn.CrossEntropyLoss()
        # predict_batch shape is [batch, classes_unnormalized_scores, seq_len]
        # target_batch shape is [batch, seq_len] (target - tensor of ints)
        L = loss_f(predict_batch, target_batch)
        return L
        