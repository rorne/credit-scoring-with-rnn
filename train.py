<<<<<<< HEAD
import os
import pandas as pd
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from credit-scoring.utils import read_parquet_dataset_from_local
from credit-scoring.dataset_preprocessing_utils import features, transform_credits_to_sequences, create_padded_buckets
from collections import defaultdict

from credit-scoring.data_generators import batches_generator
from credit-scoring.pytorch_training import train_epoch, eval_model, inference
from credit-scoring.training_aux import EarlyStopping

import subprocess
import os


def main():
    download_command = "wget https://storage.yandexcloud.net/ds-ods/files/materials/02464a6f/data_for_competition.zip -O data.zip"
    subprocess.run(download_command, shell=True, check=True)

    unzip_command = "unzip data.zip"
    os.rename('data_for_competition', 'data')
    subprocess.run(unzip_command, shell=True, check=True)
    TRAIN_DATA_PATH = "./data/train_data/"
    TEST_DATA_PATH = "./data/test_data/"

    TRAIN_TARGET_PATH = "./data/train_target.csv"

    train_target = pd.read_csv(TRAIN_TARGET_PATH)
    train_lens = []
    test_lens = []
    uniques = defaultdict(set)

    for step in range(0, 1, 1):
        credits_frame = read_parquet_dataset_from_local(TRAIN_DATA_PATH, step, 4, verbose=True)
        seq_lens = credits_frame.groupby("id").agg(seq_len=("rn", "max"))["seq_len"].values
        train_lens.extend(seq_lens)
        credits_frame.drop(columns=["id", "rn"], inplace=True)
        for feat in credits_frame.columns.values:
            uniques[feat] = uniques[feat].union(credits_frame[feat].unique())
    train_lens = np.hstack(train_lens)

    for step in range(0, 1, 1):
        credits_frame = read_parquet_dataset_from_local(TEST_DATA_PATH, step, 2, verbose=True)
        seq_lens = credits_frame.groupby("id").agg(seq_len=("rn", "max"))["seq_len"].values
        test_lens.extend(seq_lens)
        credits_frame.drop(columns=["id", "rn"], inplace=True)
        for feat in credits_frame.columns.values:
            uniques[feat] = uniques[feat].union(credits_frame[feat].unique())
    test_lens = np.hstack(test_lens)
    uniques = dict(uniques)
    keys_ = list(range(1, 59)) 
    lens_ = list(range(1, 41)) + [45] * 5 + [50] * 5 + [58] * 8
    bucket_info = dict(zip(keys_, lens_))

    train, val = train_test_split(train_target, random_state=42, test_size=0.1)
    create_buckets_from_credits(TRAIN_DATA_PATH,
                            bucket_info=bucket_info,
                            save_to_path=TRAIN_BUCKETS_PATH,
                            frame_with_ids=train,
                            num_parts_to_preprocess_at_once=4, 
                            num_parts_total=12, has_target=True)

    TRAIN_BUCKETS_PATH = "../data/train_buckets_rnn"
    VAL_BUCKETS_PATH = "../data/val_buckets_rnn"
    TEST_BUCKETS_PATH = "../data/test_buckets_rnn"
    for buckets_path in [TRAIN_BUCKETS_PATH, VAL_BUCKETS_PATH, TEST_BUCKETS_PATH]:
        os.makedirs(buckets_path, exist_ok=True)

    create_buckets_from_credits(TRAIN_DATA_PATH,
                            bucket_info=bucket_info,
                            save_to_path=VAL_BUCKETS_PATH,
                            frame_with_ids=val,
                            num_parts_to_preprocess_at_once=4, 
                            num_parts_total=12, has_target=True)
    dataset_train = sorted([os.path.join(TRAIN_BUCKETS_PATH, x) for x in os.listdir(TRAIN_BUCKETS_PATH)])

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
=======
import os
import pickle
import subprocess
import sys
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from credit_scoring.data_generators import batches_generator
from credit_scoring.dataset_preprocessing_utils import (
    create_padded_buckets,
    features,
    transform_credits_to_sequences,
)
from credit_scoring.pytorch_training import eval_model, inference, train_epoch
from credit_scoring.training_aux import EarlyStopping
from credit_scoring.utils import read_parquet_dataset_from_local


class CreditRNNModule(pl.LightningModule):
    def __init__(self, model_config, data_config, training_config):
        super(CreditRNNModule, self).__init__()
        self.model = CreditsRNN(features, model_config)
        self.lr = training_config.learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    subprocess.run("./load.sh", shell=True, check=True)

    TRAIN_DATA_PATH = cfg.data.train_path
    TEST_DATA_PATH = cfg.data.test_path
    TRAIN_TARGET_PATH = cfg.data.train_target_path

    train_target = pd.read_csv(TRAIN_TARGET_PATH)
    train_lens = []
    test_lens = []
    uniques = defaultdict(set)

    for step in range(0, 1, 1):
        credits_frame = read_parquet_dataset_from_local(
            TRAIN_DATA_PATH, step, 4, verbose=True
        )
        seq_lens = (
            credits_frame.groupby("id").agg(seq_len=("rn", "max"))["seq_len"].values
        )
        train_lens.extend(seq_lens)
        credits_frame.drop(columns=["id", "rn"], inplace=True)
        for feat in credits_frame.columns.values:
            uniques[feat] = uniques[feat].union(credits_frame[feat].unique())
    train_lens = np.hstack(train_lens)

    for step in range(0, 1, 1):
        credits_frame = read_parquet_dataset_from_local(
            TEST_DATA_PATH, step, 2, verbose=True
        )
        seq_lens = (
            credits_frame.groupby("id").agg(seq_len=("rn", "max"))["seq_len"].values
        )
        test_lens.extend(seq_lens)
        credits_frame.drop(columns=["id", "rn"], inplace=True)
        for feat in credits_frame.columns.values:
            uniques[feat] = uniques[feat].union(credits_frame[feat].unique())
    test_lens = np.hstack(test_lens)
    uniques = dict(uniques)
    keys_ = list(range(1, 59))
    lens_ = list(range(1, 41)) + [45] * 5 + [50] * 5 + [58] * 8
    bucket_info = dict(zip(keys_, lens_))

    train, val = train_test_split(train_target, random_state=42, test_size=0.1)

    TRAIN_BUCKETS_PATH = "./data/train_buckets_rnn"
    VAL_BUCKETS_PATH = "./data/val_buckets_rnn"
    TEST_BUCKETS_PATH = "./data/test_buckets_rnn"
    for buckets_path in [TRAIN_BUCKETS_PATH, VAL_BUCKETS_PATH, TEST_BUCKETS_PATH]:
        os.makedirs(buckets_path, exist_ok=True)

    create_buckets_from_credits(
        TRAIN_DATA_PATH,
        bucket_info=bucket_info,
        save_to_path=TRAIN_BUCKETS_PATH,
        frame_with_ids=train,
        num_parts_to_preprocess_at_once=4,
        num_parts_total=12,
        has_target=True,
    )

    create_buckets_from_credits(
        TRAIN_DATA_PATH,
        bucket_info=bucket_info,
        save_to_path=VAL_BUCKETS_PATH,
        frame_with_ids=val,
        num_parts_to_preprocess_at_once=4,
        num_parts_total=12,
        has_target=True,
    )
    dataset_train = sorted(
        [os.path.join(TRAIN_BUCKETS_PATH, x) for x in os.listdir(TRAIN_BUCKETS_PATH)]
    )

    dataset_val = sorted(
        [os.path.join(VAL_BUCKETS_PATH, x) for x in os.listdir(VAL_BUCKETS_PATH)]
    )
    create_buckets_from_credits(
        TEST_DATA_PATH,
        bucket_info=bucket_info,
        save_to_path=TEST_BUCKETS_PATH,
        num_parts_to_preprocess_at_once=2,
        num_parts_total=2,
    )

    dataset_test = sorted(
        [os.path.join(TEST_BUCKETS_PATH, x) for x in os.listdir(TEST_BUCKETS_PATH)]
    )
    embedding_projections = {
        feat: (max(uniq) + 1, min(600, round(1.6 * (max(uniq) + 1)) ** 0.56))
        for feat, uniq in uniques.items()
    }
    es = EarlyStopping(
        patience=cfg.training.patience,
        mode="max",
        verbose=True,
        save_path=os.path.join("checkpoints", "best_checkpoint.pt"),
        metric_name=cfg.training.metric_name,
        save_format="torch",
    )
    num_epochs = cfg.training.num_epochs
    train_batch_size = cfg.training.train_batch_size
    val_batch_size = cfg.training.val_batch_size

    model = CreditsRNN(features, embedding_projections).to(device)
    optimizer = torch.optim.Adam(
        lr=cfg.training.learning_rate, params=model.parameters()
    )

    for epoch in range(num_epochs):
        train_epoch(
            model,
            optimizer,
            dataset_train,
            batch_size=train_batch_size,
            shuffle=True,
            print_loss_every_n_batches=500,
            device=device,
        )

        val_roc_auc = eval_model(
            model, dataset_val, batch_size=val_batch_size, device=device
        )
        es(val_roc_auc, model)

        if es.early_stop:
            break
        torch.save(
            model.state_dict(),
            os.path.join("checkpoints", f"epoch_{epoch+1}_val_{val_roc_auc:.3f}.pt"),
        )

        train_roc_auc = eval_model(
            model, dataset_train, batch_size=val_batch_size, device=device
        )


if __name__ == "__main__":
    main()
>>>>>>> 3528445 (Apply formatting changes by black and isort)
