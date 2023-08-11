import argparse
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from pathlib import Path

import h5py
import numpy as np
import torch as T
from dotmap import DotMap

from src.plotting import plot_multi_hists_2, quantile_bins

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_const", type=int, default=150, help="Number of constituents to use.")
    parser.add_argument("--cond", type=str, default="uncond", help="Whether to use conditional or unconditional models.")

    parser.add_argument("--epic_jedi_project", type=str, default="final_cedric_changes", help="Name of the EPiC-JeDi project.")
    parser.add_argument("--epicjedi_directory", type=str, default="2023-07-16_09-58-26-905101", help="Name of the EPiC-JeDi directory.")

    parser.add_argument("--pc_jedi_project", type=str, default="pcjedi", help="Name of the PC-JeDi project.")
    parser.add_argument("--pcjedi_directory", type=str, default="model", help="Name of the PC-JeDi directory.")

    parser.add_argument("--epic_fm_project", type=str, default="epic_fm", help="Name of the EPiC-FM project.")
    parser.add_argument("--epic_fm_directory", type=str, default="150_uncond", help="Name of the EPiC-FM directory.")

    parser.add_argument("--epic_gan_project", type=str, default="epic_gan", help="Name of the EPiC-GAN project.")
    parser.add_argument("--epic_gan_directory", type=str, default="150", help="Name of the EPiC-GAN directory.")   

    args = parser.parse_args()
    return args


def main():

    args = get_args()

    num_const = args.num_const
    epic_jedi_project = args.epic_jedi_project
    epicjedi_directory = args.epicjedi_directory

    pc_jedi_project = args.pc_jedi_project
    pcjedi_directory = args.pcjedi_directory

    epic_fm_project = args.epic_fm_project
    epic_fm_directory = args.epic_fm_directory

    epic_gan_project = args.epic_gan_project
    epicgan_directory = args.epic_gan_directory

    cond = args.cond

    #========== CONFIGURATION ==========
    nbins = 50
    jet_types = ["t"]  # , "g", "q", "w", "z"]
    jet_type_dict = {"g":"Gluon", "q":"Quark", "t":"Top", "w":"W", "z":"Z"}
    plot_dir = (
        f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/plots/{num_const}{cond}_constituents"
    )

    eta_range = (-0.5, 0.5) if num_const == 150 else (-0.5, 0.5)
    phi_range = (-0.5, 0.5)

    if not Path(plot_dir).exists():
        Path(plot_dir).mkdir(parents=True)

    #========== DATA and directory config ==========
    if (num_const == 30):
        all_data = [
            {
                "label": "MC",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data/jetnet_data_{num_const}",
                "file": f"jetnet_data_test_csts",
                "hist_kwargs": {"color": "tab:blue", "fill": True, "alpha": 0.3},
                "err_kwargs": {"color": "tab:blue", "hatch": "///"},
            },
            {
                "label": f"EPiC-FM",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_fm_project}/{epic_fm_directory}",
                "file": "midpoint_100_csts",
                "hist_kwargs": {"color": "brown"},
            },
            {
                "label": f"EPiC-JeDi",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_jedi_project}/{epicjedi_directory}",
                "file": "midpoint_100_csts",
                "hist_kwargs": {"color": "b", "ls": "-"},
            },
            {
                "label": f"EPiC-GAN",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_gan_project}/{epicgan_directory}",
                "file": "30_csts",
                "hist_kwargs": {"color": "orange", "ls": "-"},
            },
            # {
            #     "label": f"PC-JeDi",
            #     "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{pc_jedi_project}/{pcjedi_directory}",
            #     "file": "ddim_200_csts",
            #     "hist_kwargs": {"color": "red"},
            # },
        ]
    else:
        all_data = [
            {
                "label": "MC",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data/jetnet_data_{num_const}",
                "file": f"jetnet_data_test_csts",
                "hist_kwargs": {"color": "tab:blue", "fill": True, "alpha": 0.3},
                "err_kwargs": {"color": "tab:blue", "hatch": "///"},
            },
            {
                "label": f"EPiC-FM",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_fm_project}/{epic_fm_directory}",
                "file": "midpoint_100_csts",
                "hist_kwargs": {"color": "brown"},
            },
            {
                "label": f"EPiC-JeDi",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_jedi_project}/{epicjedi_directory}",
                "file": "midpoint_100_csts",
                "hist_kwargs": {"color": "b", "ls": "-"},
            },
            {
                "label": f"EPiC-GAN",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_gan_project}/{epicgan_directory}",
                "file": "150_csts",
                "hist_kwargs": {"color": "orange", "ls": "-"},
            },
        ]
    all_data = [DotMap(**d) for d in all_data]

    # Cycle through the jet types and variables and make each plot
    for jet_type in jet_types:
        # Load the data and plot the inclusive marginalls
        for d in all_data:
            data_file = Path(d.path, "outputs", jet_type, d.file + ".h5")

            with h5py.File(data_file) as f:
                arr = f["etaphipt_frac"][:]
            arr[..., 0] = np.clip(arr[..., 0], eta_range[0], eta_range[1])
            arr[..., 1] = np.clip(arr[..., 1], phi_range[0], phi_range[1])
            d[jet_type] = arr
            d[jet_type + "mask"] = np.any(arr != 0, axis=-1)


    #========== PLOTTING ==========
        # Plot the inclusive marginals
        plot_multi_hists_2(
            data_list=[d[jet_type][d[jet_type + "mask"]] for d in all_data],
            data_labels=[d.label for d in all_data],
            col_labels=[r"$\Delta \eta$", r"$\Delta \phi$", r"$\frac{p_\mathrm{T}}{\mathrm{Jet} p_\mathrm{T}}$"],
            hist_kwargs=[d.hist_kwargs for d in all_data],
            err_kwargs=[d.err_kwargs for d in all_data],
            bins=[
                np.linspace(eta_range[0], eta_range[1], nbins),
                np.linspace(phi_range[0], phi_range[1], nbins),
                quantile_bins(d[jet_type][..., -1].flatten(), nbins),
            ],
            do_err=True,
            legend_kwargs={
                # "title": f"{jet_type_dict[jet_type]} jets",
                "alignment": "left",
                "frameon": False,
                "fontsize": 13,
            },
            rat_ylim=[0.5, 1.5],
            do_ratio_to_first=True,
            path=Path(plot_dir, f"{jet_type}_constituents.pdf"),
            do_norm=True,
            logy=False,
            ypad=[0.3,0.3,0.],

        )

        # Plot the leading three constituent pt values
        pts = [d[jet_type][..., -1] for d in all_data]
        top20 = [
                T.sort(T.tensor(pt), dim=-1, descending=True)[0][:, [0, 4, 19]]
                for pt in pts
            ]
        plot_multi_hists_2(
            data_list=top20,
            data_labels=[d.label for d in all_data],
            col_labels=[
                r"Leading constituent $\frac{p_\mathrm{T}}{\mathrm{Jet} p_\mathrm{T}}$",
                r"$5^{th}$ leading constituent $\frac{p_\mathrm{T}}{\mathrm{Jet} p_\mathrm{T}}$",
                r"$20^{th}$ leading constituent $\frac{p_\mathrm{T}}{\mathrm{Jet} p_\mathrm{T}}$",
            ],
            hist_kwargs=[d.hist_kwargs for d in all_data],
            err_kwargs=[d.err_kwargs for d in all_data],
            bins=quantile_bins(top20[0], nbins, axis=(0)).T.tolist(),
            do_err=True,
            legend_kwargs={
                # "title": f"{jet_type_dict[jet_type]} jets",
                "alignment": "left",
                "frameon": False,
                "fontsize": 13,
            },
            rat_ylim=[0.5, 1.5],
            do_ratio_to_first=True,
            path=Path(plot_dir, f"{jet_type}_leading_constituents.pdf"),
            do_norm=True,
            logy=False,
            ypad=[.15,.15,.3],
        )

if __name__ == "__main__":
    main()