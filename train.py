import os
import pandas as pd
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
import tqdm
from credit-scoring/utils import read_parquet_dataset_from_local
from credit-scoring/dataset_preprocessing_utils import features, transform_credits_to_sequences, create_padded_buckets
from collections import defaultdict

from credit-scoring/data_generators import batches_generator
from credit-scoring/pytorch_training import train_epoch, eval_model, inference
from credit-scoring/training_aux import EarlyStopping

def main():

    TRAIN_BUCKETS_PATH = "../data/train_buckets_rnn"
    VAL_BUCKETS_PATH = "../data/val_buckets_rnn"
    TEST_BUCKETS_PATH = "../data/test_buckets_rnn"

    train, val = train_test_split(train_target, random_state=42, test_size=0.1)
    create_buckets_from_credits(TRAIN_DATA_PATH,
                            bucket_info=bucket_info,
                            save_to_path=TRAIN_BUCKETS_PATH,
                            frame_with_ids=train,
                            num_parts_to_preprocess_at_once=4, 
                            num_parts_total=12, has_target=True)

    dataset_train = sorted([os.path.join(TRAIN_BUCKETS_PATH, x) for x in os.listdir(TRAIN_BUCKETS_PATH)])
    create_buckets_from_credits(TRAIN_DATA_PATH,
                            bucket_info=bucket_info,
                            save_to_path=VAL_BUCKETS_PATH,
                            frame_with_ids=val,
                            num_parts_to_preprocess_at_once=4, 
                            num_parts_total=12, has_target=True)

    dataset_val = sorted([os.path.join(VAL_BUCKETS_PATH, x) for x in os.listdir(VAL_BUCKETS_PATH)])
    create_buckets_from_credits(TEST_DATA_PATH,
                            bucket_info=bucket_info,
                            save_to_path=TEST_BUCKETS_PATH, num_parts_to_preprocess_at_once=2, 
                            num_parts_total=2)

    dataset_test = sorted([os.path.join(TEST_BUCKETS_PATH, x) for x in os.listdir(TEST_BUCKETS_PATH)])
    embedding_projections = {feat: (max(uniq)+1, min(600, round(1.6 * (max(uniq)+1))**0.56)) for feat, uniq in uniques.items()}
    es = EarlyStopping(patience=3, mode="max", verbose=True, save_path=os.path.join(path_to_checkpoints, "best_checkpoint.pt"), 
                   metric_name="ROC-AUC", save_format="torch")
    num_epochs = 10
    train_batch_size = 128
    val_batch_size = 128

    model = CreditsRNN(features, embedding_projections).to(device)
    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())


    for epoch in range(num_epochs):
    	train_epoch(model, optimizer, dataset_train, batch_size=train_batch_size, 
                shuffle=True, print_loss_every_n_batches=500, device=device)
    
        val_roc_auc = eval_model(model, dataset_val, batch_size=val_batch_size, device=device)
        es(val_roc_auc, model)
    
        if es.early_stop:
	    break
        torch.save(model.state_dict(), os.path.join(path_to_checkpoints, f"epoch_{epoch+1}_val_{val_roc_auc:.3f}.pt"))
        
        train_roc_auc = eval_model(model, dataset_train, batch_size=val_batch_size, device=device)

if __name__ == "__main__":
    main()