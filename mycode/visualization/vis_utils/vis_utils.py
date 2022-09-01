import argparse


__all__ = [
    "get_parser_args",
]


def get_parser_args(default_path):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config-file", type=str, default="configs/DA_development.yaml",
        help="Path to the dataset."
    )
    
    parser.add_argument(
        "-d", "--dataset-path", default=default_path,
        help="Path to the dataset."
    )

    parser.add_argument(
        "-n", "--sample-amount", type=int, default=20,
        help="Number of random samples."
    )

    parser.add_argument(
        "--use-cgnet", action="store_true",
    )

    parser.add_argument(
        "-r", "--random-seed", type=int, default=None,
    )

    return parser.parse_args()
