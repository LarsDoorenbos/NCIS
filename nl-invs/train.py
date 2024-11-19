
import os
import yaml

import argparse

import nl_invariants


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--e', type=int, default=-1)
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--pca_var', type=float, default=-1)
    parser.add_argument('--dataset', type=str, default='null')
    args = parser.parse_args()

    with open("nl-invs/params.yml", 'r') as f:
        params = yaml.safe_load(f)

    params["batch_size"] = args.bs if args.bs != -1 else params["batch_size"]
    params["optim"]["learning_rate"] = args.lr if args.lr != -1 else params["optim"]["learning_rate"]
    params["pca_variance_percentage"] = args.pca_var if args.pca_var != -1 else params["pca_variance_percentage"]
    params["dataset_file"] =  args.dataset if args.dataset != 'null' else params["dataset_file"]
    params["max_epochs"] = args.e if args.e != -1 else params["max_epochs"]
    
    # Remove SLURM_JOBID to prevent ignite assume we are using SLURM to run multiple tasks.
    os.environ.pop("SLURM_JOBID", None)

    # Run in a single node
    nl_invariants.run_train(params)


if __name__ == "__main__":
    main()
