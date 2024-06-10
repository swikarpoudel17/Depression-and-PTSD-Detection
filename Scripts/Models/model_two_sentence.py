#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: spoudel
"""

import torch
import torch.nn as nn
import re

class TwoSentenceAttention(nn.Module):
    def __init__(self, checkpoint, device, cls_embeds_size=768, n_heads=4, output_dim=3):
        super(TwoSentenceAttention, self).__init__()
        self.mha = nn.MultiheadAttention(cls_embeds_size, n_heads, batch_first=True)
        self.fc1 = nn.Linear(cls_embeds_size, 10)
        self.fc2 = nn.Linear(10, output_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def cleanTweet(self, text):
        cleanText = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        return cleanText

    def get_embedding(self, text, tokenizer, transformer, device):
        current_tweet = text
        inputs = tokenizer(self.cleanTweet(str(current_tweet)), return_tensors="pt", padding=True, truncation=True).to(device)
        output = transformer(**inputs, output_hidden_states=True)
        last_hidden_state = output.hidden_states[-1]
        cls_emb = last_hidden_state[0,0,:]
        return cls_emb

    def forward(self, tweets, tokenizer, transformer, device):
        for t_idx, t in enumerate(tweets[:-1]):
            current_sentence_seq = tweets[t_idx]['text'] + ' ' + tweets[t_idx+1]['text']
            cls_embedding_input = self.get_embedding(current_sentence_seq, tokenizer, transformer, device)
            if t_idx == 0:
                tensor_of_embeds = cls_embedding_input.unsqueeze(0)
            else:
                tensor_of_embeds = torch.cat((tensor_of_embeds, cls_embedding_input.unsqueeze(0)), dim=0)
        
        mha_input = tensor_of_embeds.unsqueeze(0)
        mha_output, _ = self.mha(mha_input, mha_input, mha_input)        
        mha_output = mha_output.squeeze(0)
        output_mean = torch.mean(mha_output, dim=0, keepdim=True)
        dense_outputs = self.fc1(output_mean)
        dense_outputs = self.tanh(dense_outputs)
        dense_outputs = self.fc2(dense_outputs)
        dense_outputs = self.tanh(dense_outputs)
        outputs = self.softmax(dense_outputs)                              
        return outputs
