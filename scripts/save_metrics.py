import pyrootutils
import yaml

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import argparse
from argparse import Namespace
from pathlib import Path
from src.evaluation_utils import dump_metrics, get_output_file_list


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data",
        help="Path to directory where all the data is saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to evaluate.",
        default="jetnet_data_150",
    )
    parser.add_argument(
        "--jet_types",
        type=str,
        help="Comma separated names of the jet types to load from.",
        default="t",
    )
    parser.add_argument(
        "--file_names",
        type=str,
        default="jetnet_data_train_csts.h5",
        help="Comma separated names of the files to generate from.",
    )

    parser.add_argument(
        "--num_eval_samples",
        type=int,
        help="Number of samples per batch in the bootsrapped evaluation methods.",
        default=50000,
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        help="Number of bootstrapped batches to evaluate.",
        default=40,
    )
    parser.add_argument(
        "--numparticles",
        type=int,
        default=150,
        help="Number of particles in the jet.",
    )

    parser.add_argument(
        "--realname_modelname_filenames",
        type=str,
        default="",
        help="Combined argument which takes precidence over model_name and file_names.",
    )
    parser.add_argument("--key", type=str, default="etaphipt_frac", help="key to use for the data in the h5 file.")
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    # Check if the combined argument is defined, this takes precidence
    if args.realname_modelname_filenames != "":
        (
            args.real_name,
            args.model_name,
            args.file_names,
        ) = args.realname_modelname_filenames.split("/")

    # Paths to the relevant folders
    print(f"Evaluating Network: {args.model_name}\n")
    print(f"- Jet Types: {args.jet_types}")
    print(f"- File search: {args.file_names}")
    path = Path(args.save_dir, args.model_name, "outputs")
    
    #laod the full config to read the num_particles
    try:
        with open(args.save_dir + "/" + args.model_name + "/full_config.yaml") as file:
            info = yaml.safe_load(file)
        num_particles = info["datamodule"]["data_conf"]["jetnet_config"]["num_particles"]
    except FileNotFoundError:
        print("No full config found, using default value of 30")
        num_particles = args.numparticles

    # Get all the files to run over, allow wildcarding
    file_paths = get_output_file_list(path, args.jet_types, args.file_names)

    # Cycle through the files and calculate the metrics
    for file_path in file_paths:
        print(f"-- running on {file_path.parent.name} jets from {file_path.name}")

        # Create the output file name by swapping the csts flag with scores
        outpath = file_path.parent / file_path.stem.replace("csts", "scores")
        outpath = outpath.with_suffix(".yaml")

        # Also get the path to the real data for calculating metrics
        jet_type = file_path.parent.name
        real_name = f"jetnet_data_{num_particles}"

        jetnetpath = Path("/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data/")
        real_path = Path(
            jetnetpath,
            real_name,
            "outputs",
            jet_type,
            "jetnet_data_test_csts.h5",
        )

        # if file exists, skip
        # if outpath.is_file():
        #     continue

        # Save the metrics
        key = "substructure_frac"
        try:
            dump_metrics(
                file_path,
                outpath,
                real_path,
                jet_type,
                args.num_eval_samples,
                args.num_batches,
                key
            )
            print("--- done")
        except Exception as e:
            print("--- file failed, read the error below:")
            print(e)


if __name__ == "__main__":
    main()
