#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: spoudel
"""

import torch
import mlflow
import torcheval.metrics.functional as f


def calculate_accuracy(model, criterion, tweets_train, tweets_val, conditions_labels, val_conditions_labels, output_dim, device, epoch, log_file_name, val_batch_counter, tokenizer, transformer):
    pred_list = []
    model.eval()
    for idx, u in enumerate(tweets_train):  
        tweets_len = min(1000, len(u))
        u = u[-tweets_len:]
        with torch.no_grad():
            predictions = model(u, tokenizer, transformer, device)
            pred_list.append(predictions.argmax())
            update = {"Idx": idx+1, "Training predictions": predictions, "Training output": predictions.argmax(), "Actuals": conditions_labels[idx]}
            print(update)
            log_file = open(log_file_name, "a")
            log_file.write(str(update) + "\n")
            log_file.close()

    preds = torch.stack(pred_list)
    train_f1 = f.multiclass_f1_score(preds, conditions_labels, num_classes=output_dim)
    train_precision = f.multiclass_precision(preds, conditions_labels, num_classes=output_dim)
    train_recall = f.multiclass_recall(preds, conditions_labels, num_classes=output_dim)
    train_accuracy = f.multiclass_accuracy(preds, conditions_labels, num_classes=output_dim)
    mlflow.log_metric("train_f1", train_f1, step=epoch+1)
    mlflow.log_metric("train_precision", train_precision, step=epoch+1)
    mlflow.log_metric("train_recall", train_recall, step=epoch+1)
    mlflow.log_metric("train_accuracy", train_accuracy, step=epoch+1)
    
    pred_list = []
    model.eval()
    val_loss_all = 0.0
    for idx, u in enumerate(tweets_val):
        tweets_len = min(1000, len(u))
        u = u[-tweets_len:]
        with torch.no_grad():
            predictions = model(u, tokenizer, transformer, device)
            pred_list.append(predictions.argmax())
            update = {"Idx":idx+1, "Val predictions": predictions , "Val output":predictions.argmax(), "Actuals": val_conditions_labels[idx]}
            print(update)
            log_file = open(log_file_name, "a")
            log_file.write(str(update) + "\n")
            log_file.close()
            val_loss = criterion(predictions.view(1,3), val_conditions_labels[idx].view(-1))
            val_loss_all += val_loss.item()
            mlflow.log_metric("Batch val loss", val_loss.item(), step=val_batch_counter)
            val_batch_counter = val_batch_counter + 1
    
    average_val_loss = val_loss_all / len(tweets_val)
    mlflow.log_metric("val_loss", average_val_loss, step=epoch+1)
    
    preds = torch.stack(pred_list)
    val_f1 = f.multiclass_f1_score(preds, val_conditions_labels, num_classes=output_dim)
    val_precision = f.multiclass_precision(preds, val_conditions_labels, num_classes=output_dim)
    val_recall = f.multiclass_recall(preds, val_conditions_labels, num_classes=output_dim)
    val_accuracy = f.multiclass_accuracy(preds, val_conditions_labels, num_classes=output_dim)
    mlflow.log_metric("val_f1", val_f1, step=epoch+1)
    mlflow.log_metric("val_precision", val_precision, step=epoch+1)
    mlflow.log_metric("val_recall", val_recall, step=epoch+1)
    mlflow.log_metric("val_accuracy", val_accuracy, step=epoch+1)
    
    return (train_accuracy, val_accuracy, average_val_loss, val_batch_counter)
