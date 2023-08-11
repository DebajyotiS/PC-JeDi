import pyrootutils
import yaml

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import argparse
from argparse import Namespace
from pathlib import Path

from src.evaluation_utils import get_output_file_list
from src.jet_substructure import dump_hlvs
import numpy as np
import h5py

def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/users/s/senguptd/scratch/jet_diffusion/epic_gan",
        help="Path to directory where all the data is saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="150",
        help="Name of the model to evaluate.",
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
        default="*_csts.h5",
        help="Comma separated names of the files to generate from.",
    )
    parser.add_argument(
        "--realname_modelname_filenames",
        type=str,
        default="",
        help="Combined argument which takes precidence over model_name and file_names.",)
    
    parser.add_argument(
        "--key",
        type=str,
        default="etaphipt_frac",
        help="Key to use for substructure.",
    )
    args = parser.parse_args()
    return args

def get_particle_count(file_path: Path) -> int:
    try:
        with h5py.File(file_path, "r") as f:
            count = f["etaphipt_frac"].shape[1]
    except FileNotFoundError:
        # The folder nanme is the number of particles
        name = file_path.parent.name
        if "cond" in name:
            name = name.split("_")[0]
        count = int(name)
    return count


def main() -> None:
    args = get_args()
    keytouse = args.key
    
    # Check if the combined argument is defined, this takes precidence
    if args.realname_modelname_filenames != "":
        _, args.model_name, args.file_names = args.realname_modelname_filenames.split(
            "/"
        )

    # Paths to the relevant folders
    print(f"Evaluating Network: {args.model_name}\n")
    print(f"- Jet Types: {args.jet_types}")
    print(f"- File search: {args.file_names}")
    path = Path(args.save_dir, args.model_name, "outputs")

    # Get all the files to run over, allow wildcarding
    file_paths = get_output_file_list(path, args.jet_types, args.file_names)
    num_particles = get_particle_count(file_paths[0])

    eta_range = [-0.999, 0.94] if num_particles == 30 else [-1.6, 1.0]
    # Cycle through the files and calculate the substructure
    for file_path in file_paths:
        print(f"-- running on {file_path.parent.name} jets from {file_path.name}")
        print(f"-- using key {keytouse}")

        # Create the output file name by swapping the csts flag with substructure
        # load the constituents from the file
        with h5py.File(file_path, "r") as f:
            gen_nodes = f[f"{keytouse}"][:]
        gen_nodes[..., 0] = np.clip(gen_nodes[..., 0], eta_range[0], eta_range[1])
        gen_nodes[..., 1] = np.clip(gen_nodes[..., 1], -0.5,0.5)

        append = "_frac" if keytouse == "etaphipt_frac" else ""
        outpath = file_path.parent / file_path.stem.replace("csts", f"substructure{append}")
        outpath = outpath.with_suffix(f".h5")

        # If the file already exists, skip it
        if outpath.exists():
            print("--- file already exists, skipping")
            continue

        # Save the substructure variables in the same folder with an added suffix
        try:
            dump_hlvs(gen_nodes, outpath, plot=True)
            print("--- done")
        except Exception as e:
            print("--- file failed, read the error below:")
            print(e)

    print("Finished all tasks")


if __name__ == "__main__":
    main()
