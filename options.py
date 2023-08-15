import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## mode ########################
    parser.add_argument("--simplified", default=False, action='store_true')

    args = parser.parse_args()

    return args