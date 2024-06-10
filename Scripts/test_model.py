#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: spoudel
"""

import torch
import sys
from data_loader import load_test_data, encode_test_labels
from Models.model_lstm import LSTMOne
from Models.model_lstm_2layer import LSTMTwo
from Models.model_attention import Attention
from Models.model_temporal import Temporal
from Models.model_two_sentence import TwoSentenceAttention
from transformers import AutoTokenizer, RobertaForSequenceClassification


def get_plm(checkpoint, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    transformer = RobertaForSequenceClassification.from_pretrained(checkpoint)
    transformer = transformer.to(device)
    return (tokenizer, transformer)

def test_model(model_name, test_path, checkpoint):
    SEED = 11
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_dim = 768
    hidden_dim = 50
    output_dim = 3
    cls_embeds_size = 768
    n_heads = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tweets_test, conditions_test = load_test_data(test_path)
    conditions_labels_test = encode_test_labels(conditions_test, device)
    
    tokenizer, transformer = get_plm(checkpoint, device)

    if model_name == "LSTMOne":
        model = LSTMOne(checkpoint, device).to(device)
        model.load_state_dict(torch.load('../Trained_models/LSTMOneTrained/BestModel'))
    elif model_name == "LSTMTwo":
        model = LSTMTwo(checkpoint, device).to(device)
        model.load_state_dict(torch.load('../Trained_models/LSTMTwo/BestModel'))
    elif model_name == "Attention":
        model = TransformAttention(checkpoint, device).to(device)
        model.load_state_dict(torch.load('../Trained_models/Attention/BestModel'))
    elif model_name == "Temporal":
        model = TransformAttention(checkpoint, device).to(device)
        model.load_state_dict(torch.load('../Trained_models/Temporal/BestModel'))
    elif model_name == "TwoSentenceAttention":
        model = TransformAttention(checkpoint, device).to(device)
        model.load_state_dict(torch.load('../Trained_models/TwoSentenceAttention/BestModel'))
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.eval()
    correct_predictions = 0
    with torch.no_grad():
        for idx, u in enumerate(tweets_test):
            tweets_len = min(1000, len(u))
            u = u[-tweets_len:]
            predictions = model(u, tokenizer, transformer, device)
            predicted_label = predictions.argmax()
            correct_predictions += (predicted_label == conditions_labels_test[idx]).sum().item()

    accuracy = correct_predictions / len(tweets_test)
    print(f'Test Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    test_path = '../Data/3_testing_set.pickle'
    checkpoint = 'cardiffnlp/twitter-roberta-base-emotion'
    test_model(model_name, test_path, checkpoint)
