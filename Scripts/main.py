#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: spoudel
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Train or test a model.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--model', type=str, required=True, choices=['LSTMOne', 'LSTMTwo', 'Attention', 'Temporal',  'TwoSentenceAttention'], help='Model name: LSTMOne, LSTMTwo, Attention, Temporal or TwoSentenceAttention')
    # parser.add_argument('--log', type=str, help='Log file name for training')
    args = parser.parse_args()

    if args.mode == "train":
        ## Creating folder for saving trained models
        os.makedirs(f'../Trained_models/{args.model}', exist_ok=True)
        
        ## Creating log file for saving all necessary logs
        log_file = open(f'../Logs/{args.model}' , mode='a', encoding='utf-8')
        log_file.close()
        
        os.system(f"python3 ../Scripts/train_model.py {args.model} ../Logs/{args.model}")
        
    elif args.mode == "test":
        os.system(f"python3 ../Scripts/test_model.py {args.model}")

if __name__ == "__main__":
    main()
