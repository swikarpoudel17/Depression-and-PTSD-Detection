#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: spoudel
"""

import torch
import torch.nn as nn
import re

class LSTMTwo(nn.Module):
    def __init__(self, checkpoint, device, input_dim=768, hidden_dim=50, output_dim=3):
        super(LSTMTwo, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.fc1 = nn.Linear(hidden_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.hidden_dim = hidden_dim
        
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

    def forward(self, texts, tokenizer, transformer, device):
        h0 = torch.zeros(1, self.hidden_dim).unsqueeze(1).to(device)
        c0 = torch.zeros(1, self.hidden_dim).unsqueeze(1).to(device)
        for i in texts:
            cls_embedding_input = self.get_embedding(i['text'], tokenizer, transformer, device)
            cls_embedding_input = cls_embedding_input.unsqueeze(0).unsqueeze(0)
            output, (h0, c0) = self.lstm(cls_embedding_input, (h0, c0))
        dense_outputs = self.fc1(output)
        dense_outputs = self.tanh(dense_outputs)
        dense_outputs = self.fc2(dense_outputs)
        dense_outputs = self.tanh(dense_outputs)
        dense_outputs_squeezed = dense_outputs.squeeze(0).squeeze(0)
        outputs = self.softmax(dense_outputs_squeezed)
        return outputs
