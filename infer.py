def main():

    path_to_checkpoints = "./checkpoints/pytorch_baseline/"
    model.load_state_dict(torch.load(os.path.join(path_to_checkpoints, "best_checkpoint.pt")))
    test_preds = inference(model, dataset_test, batch_size=128, device=device)
    test_preds.to_csv("torch_submission.csv", index=None)

if __name__ == "__main__":
    main()