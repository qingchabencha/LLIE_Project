import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument(
        "--model_option",
        type=str,
        default="./Options/ReinexFormer_original_LOL_v1.yml",
        help="Path to the model option file",
    )
    parser.add_argument(
        "--train_save_dir",
        type=str,
        default="3",
        help="Directory to save training results, default save to './train_results', if specified the file directory, like 1, it will try to load the trained model first",
    )


    return parser.parse_args()