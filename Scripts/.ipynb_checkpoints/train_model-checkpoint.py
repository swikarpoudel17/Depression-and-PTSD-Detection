#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: spoudel
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from data_loader import load_data, encode_labels, compute_class_weights
from Models.model_lstm import LSTMOne
from Models.model_lstm_2layer import LSTMTwo
from Models.model_attention import Attention
from Models.model_temporal import Temporal
from Models.model_two_sentence import TwoSentenceAttention

from calculate_accuracy import calculate_accuracy
import sys
import logging
import os
import mlflow


from transformers import AutoTokenizer, RobertaForSequenceClassification


previous_train_loss = 9999999

mlflow.set_tracking_uri(uri="http://127.0.0.1:1995")
mlflow.set_experiment("lstm_git_try")
mlflow.start_run(nested=True, run_name= "first_run")

no_of_epochs = 10


def get_plm(checkpoint, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    transformer = RobertaForSequenceClassification.from_pretrained(checkpoint)
    transformer = transformer.to(device)
    return (tokenizer, transformer)

def train(model_name, train_path, val_path, checkpoint, log_file_name):
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

    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    logger = logging.getLogger()

    tweets_train, tweets_val, conditions_train, conditions_val = load_data(train_path,val_path)
    conditions_labels_train, conditions_labels_val = encode_labels(conditions_train, conditions_val, device)    
    class_weights_normalized= compute_class_weights(conditions_train)
    
    if model_name == "LSTMOne":
        model = LSTMOne(checkpoint, device).to(device)
    elif model_name == "LSTMTwo":
        model = LSTMTwo(checkpoint, device).to(device)
    elif model_name == "Attention":
        model = Attention(checkpoint, device).to(device)
    elif model_name == "Temporal":
        model = Temporal(checkpoint, device).to(device)
    elif model_name == "TwoSentenceAttention":
        model = TwoSentenceAttention(checkpoint, device).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    criterion = CrossEntropyLoss(weight=torch.tensor(class_weights_normalized)).to(device)
    optimizer = optim.AdamW(model.parameters())
    
    tokenizer, transformer = get_plm(checkpoint, device)

    training_batch_counter = 1
    val_batch_counter = 1
    
    best_val_acc = -9999999999999
    best_epoch = 0

    for epoch in range(no_of_epochs):
        epoch_loss = 0.0
        model.train()
        for idx, u in enumerate(tweets_train):
            optimizer.zero_grad()
            ## Taking only last 1000 tweets
            tweets_len = min(1000, len(u))
            u = u[-tweets_len:]
            predictions = model(u, tokenizer, transformer, device)
            loss = criterion(predictions.view(1,3), conditions_labels_train[idx].view(-1))
            loss.backward()
            optimizer.step()

            # Saving losses
            update = {"Idx": idx+1, "Loss": loss.item()} 
            log_file = open(log_file_name, "a")
            log_file.write(str(update) + "\n")
            log_file.close()
            print(update)

            mlflow.log_metric("Batch train loss", loss.item(), step=training_batch_counter)
            training_batch_counter = training_batch_counter + 1
            epoch_loss += loss.item()
            del predictions
            # torch.cuda.empty_cache()
        

        average_train_loss = epoch_loss / len(tweets_train)
        print(f'\t Epoch loss: {average_train_loss:.3f}')

        mlflow.log_metric("train_loss", average_train_loss, step=epoch+1)
        
        model_path = f"../Trained_models/{model_name}/" 

        save_path = model_path + 'Epoch_' + str(epoch+1)
        torch.save(model.state_dict(), save_path)

        # model_save_name = "lstm_git_try" + str(epoch+1)
        # model_info = mlflow.pytorch.log_model(pytorch_model = model,
        #                                   artifact_path = "model",
        #                                   registered_model_name = model_save_name)

        train_acc, val_acc, average_val_loss, val_batch_counter = calculate_accuracy(model, criterion, tweets_train, tweets_val, conditions_labels_train, conditions_labels_val, output_dim, device, epoch, log_file_name, val_batch_counter, tokenizer, transformer)
        
        update = {"Epoch": epoch+1, "Training Epoch loss": average_train_loss, "Validation Epoch loss": average_val_loss, "Train acc": train_acc, "Val acc": val_acc}    
        print(update)
        log_file = open(log_file_name, "a")
        log_file.write(str(update) + "\n")
        log_file.close
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model_save_path = f"../Trained_models/{model_name}/BestModel"
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Best model saved at epoch {epoch} with val accuracy {val_acc:.4f}')

        # if average_train_loss < previous_train_loss:
        #     previous_train_loss = average_train_loss
        # else:
        #     break

    print("Epoch ended")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <model_name> <log_file>")
        sys.exit(1)

    model_name = sys.argv[1]
    log_file = sys.argv[2]
    train_path = '../Data/final_training_tweets_filtered.pickle'
    val_path = '../Data/final_testing_tweets_filtered.pickle'
    # train_path = '../Data/3_training_set.pickle'
    # val_path = '../Data/3_testing_set.pickle'
    checkpoint = 'cardiffnlp/twitter-roberta-base-emotion'
    train(model_name, train_path, val_path, checkpoint, log_file)
