import torch
<<<<<<< HEAD
from credit-scoring.pytorch_training import inference
from credit-scoring.preprocess import create_buckets_from_credits
=======

from credit_scoring.preprocess import create_buckets_from_credits
from credit_scoring.pytorch_training import inference

>>>>>>> 3528445 (Apply formatting changes by black and isort)

def main():
    TEST_DATA_PATH = "./data/test_data/"
    TEST_BUCKETS_PATH = "./data/test_buckets_rnn"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_buckets_from_credits(
        TEST_DATA_PATH,
        bucket_info=bucket_info,
        save_to_path=TEST_BUCKETS_PATH,
        num_parts_to_preprocess_at_once=12,
        num_parts_total=4,
    )

    dataset_test = sorted(
        [os.path.join(TEST_BUCKETS_PATH, x) for x in os.listdir(TEST_BUCKETS_PATH)]
    )
    path_to_checkpoints = "./checkpoints/pytorch_baseline/"
    model.load_state_dict(
        torch.load(os.path.join(path_to_checkpoints, "best_checkpoint.pt"))
    )
    test_preds = inference(model, dataset_test, batch_size=128, device=device)
    test_preds.to_csv("torch_submission.csv", index=None)


if __name__ == "__main__":
    main()
