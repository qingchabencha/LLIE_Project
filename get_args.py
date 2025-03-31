import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument(
        "--model_option",
        type=str,
        default="./Options/ReinexFormer_original_LOL_v1.yml",
        help="Path to the model option file",
    )


    return parser.parse_args()