
import yaml
import argparse

from evaluation import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=float, default=-1)
    parser.add_argument('--dataset', type=str, default='null')
    parser.add_argument('--folder', type=str, default='null')
    args = parser.parse_args()

    with open("nl-invs/params.yml", 'r') as f:
        params = yaml.safe_load(f)

    params["pca_variance_percentage"] = args.p if args.p != -1 else params["pca_variance_percentage"]
    params["dataset_file"] =  args.dataset if args.dataset != 'null' else params["dataset_file"]
    params["load_from"] = args.folder if args.folder != 'null' else params["load_from"]

    # Run in a single node
    run_eval(0, params)


if __name__ == "__main__":
    main()
