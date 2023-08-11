import argparse
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from pathlib import Path
import h5py
import numpy as np
from dotmap import DotMap


from src.plotting import plot_multi_correlations, plot_multi_hists_2



jet_types = ["t"]  # , "g", "q", "w", "z"]
sub_vars = [
    # "tau1",
    # "tau2",
    # "tau3",
    "tau21",
    "tau32",
    # "d12",
    # "d23",
    # "ecf2",
    # "ecf3",
    "d2",
    "mass",
    "pt",
]

sub_vars_dict = {
    "tau1": r"$\tau_1^{\mathrm{rel}}$",
    "tau2": r"$\tau_2^{\mathrm{rel}}$",
    "tau3": r"$\tau_3^{\mathrm{rel}}$",
    "tau21": r"$\tau_{21}^{\mathrm{rel}}$",
    "tau32": r"$\tau_{32}^{\mathrm{rel}}$",
    "d12": r"$d_{12}^{\mathrm{rel}}$",
    "d23": r"$d_{23}^{\mathrm{rel}}$",
    "ecf2": r"$ECF2^{\mathrm{rel}}$",
    "ecf3": r"$ECF3^{\mathrm{rel}}$",
    "d2": r"$D_2^{\mathrm{rel}}$",
    "mass": r"$m_{J}^{\mathrm{rel}}$",
    "pt": r"$p_{\mathrm{T}}^{\mathrm{rel}}$",
}

feat_spread_vars = ["tau21", "tau32", "d2", "mass"]
feat_subvars_dict = [r"$\tau_{21}^{\mathrm{rel}}$", r"$\tau_{32}^{\mathrm{rel}}$", r"$D_2^{\mathrm{rel}}$", r"$m_{J}^{\mathrm{rel}}$"]

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_const", type=int, default=30)
    parser.add_argument("--epic_project", type=str, default="final_cedric_changes")
    parser.add_argument("--epicjedi_directory", type=str, default="2023-07-16_09-58-26-913806")
    parser.add_argument("--pc_jedi_project", type=str, default="pcjedi")
    parser.add_argument("--pcjedi_directory", type=str, default="model")
    parser.add_argument("--epicfm_project", type=str, default="epic_fm")
    parser.add_argument("--epic_fm_directory", type=str, default="30_uncond")
    parser.add_argument("--cond", type=str, default="uncond")
    parser.add_argument("--kde_points", type=int, default=5)
    parser.add_argument("--epic_gan_project", type=str, default="epic_gan", help="Name of the EPiC-GAN project.")
    parser.add_argument("--epic_gan_directory", type=str, default="30", help="Name of the EPiC-GAN directory.")  
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    num_const = args.num_const
    epic_jedi_project = args.epic_project
    epicjedi_directory = args.epicjedi_directory

    pc_jedi_project = args.pc_jedi_project
    pcjedi_directory = args.pcjedi_directory

    epic_fm_project = args.epicfm_project
    epic_fm_directory = args.epic_fm_directory
    
    epic_gan_project = args.epic_gan_project
    epicgan_directory = args.epic_gan_directory
    cond=args.cond

    end = 0.95 if num_const == 150 else 0.6
    nbins = 50
    n_kde_points = args.kde_points
    
    
    jet_types = ["t"]  # , "g", "q", "w", "z"]
    plot_dir = (
        f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/plots/{num_const}{cond}_constituents"
    )
    
    if not Path(plot_dir).exists():
        Path(plot_dir).mkdir(parents=True)
    
    
    if num_const == 30:
        all_data = [
            {
                "label": "MC",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data/jetnet_data_{num_const}",
                "file": f"jetnet_data_test",
                "hist_kwargs": {"color": "tab:blue", "fill": True, "alpha": 0.3},
                "err_kwargs": {"color": "tab:blue", "hatch": "///"},
            },
            {
                "label": f"EPiC-FM",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_fm_project}/{epic_fm_directory}",
                "file": "midpoint_100",
                "hist_kwargs": {"color": "brown"},
            },
            {
                "label": f"EPiC-JeDi",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_jedi_project}/{epicjedi_directory}",
                "file": "midpoint_100",
                "hist_kwargs": {"color": "b", "ls": "-"},
            },
            {
                "label": f"EPiC-GAN",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_gan_project}/{epicgan_directory}",
                "file": "30",
                "hist_kwargs": {"color": "orange", "ls": "-"},
            },
        ]
    else:
        all_data = [
            {
                "label": "MC",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data/jetnet_data_{num_const}",
                "file": f"jetnet_data_test",
                "hist_kwargs": {"color": "tab:blue", "fill": True, "alpha": 0.3},
                "err_kwargs": {"color": "tab:blue", "hatch": "///"},
            },
            {
                "label": f"EPiC-FM",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_fm_project}/{epic_fm_directory}",
                "file": "midpoint_100",
                "hist_kwargs": {"color": "brown"},
            },
            {
                "label": f"EPiC-JeDi",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_jedi_project}/{epicjedi_directory}",
                "file": "midpoint_100",
                "hist_kwargs": {"color": "b", "ls": "-"},
            },
            {
                "label": f"EPiC-GAN",
                "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_gan_project}/{epicgan_directory}",
                "file": "150",
                "hist_kwargs": {"color": "orange", "ls": "-"},
            },
        ]
    all_data = [DotMap(**d) for d in all_data]
    
    jet_type_dict = {"g":"Gluon", "q":"Quark", "t":"Top", "w":"W", "z":"Z"}
    
    # Cycle through the jet types and variables and make each plot
    for jet_type in jet_types:
        for sub_var in sub_vars:
            sub_data = []
            for d in all_data:
                data_file = Path(d.path, "outputs", jet_type, d.file + "_substructure_frac.h5")
                with h5py.File(data_file) as f:
                    sub_data.append(f[sub_var][:][:, None])
            bin_set = np.linspace(*np.quantile(sub_data[0], [0.001, 0.999]), nbins)
            if sub_var == "pt":
                bin_set = np.linspace(end, 1.05, nbins)
            plot_multi_hists_2(
                data_list=sub_data,
                data_labels=[d.label for d in all_data],
                col_labels=[sub_vars_dict[sub_var]],
                hist_kwargs=[d.hist_kwargs for d in all_data],
                err_kwargs=[d.err_kwargs for d in all_data],
                bins=bin_set,
                do_err=True,
                legend_kwargs={
                    # "title": f"{jet_type_dict[jet_type]} jets",
                    "alignment": "left",
                    "frameon": False,
                    "fontsize": 14,
                },
                do_ratio_to_first=True,
                path=Path(plot_dir, f"{jet_type}_{sub_var}.pdf"),
                do_norm=True,
                ypad=[0.35]
            )
    
        for d in all_data:
            # Load the requested substructure variables
            data_file = Path(d.path, "outputs", jet_type, d.file + "_substructure_frac.h5")
            with h5py.File(data_file) as f:
                for s in feat_spread_vars:
                    d[s] = f[s][:]
    
        # Combine the columns to pass to the plotter
        epic_fm = (all_data[0],all_data[1])
        epic_jedi = (all_data[0],all_data[2])
        plot_multi_correlations(
            data_list=[np.stack([d[s] for s in feat_spread_vars]).T for d in epic_jedi],
            data_labels=[d.label for d in epic_jedi],
            col_labels=feat_subvars_dict,
            n_bins=nbins,
            n_kde_points=n_kde_points,
            hist_kwargs=[d.hist_kwargs for d in epic_jedi],
            err_kwargs=[d.err_kwargs for d in epic_jedi],
            legend_kwargs={
                "loc": "upper right",
                "alignment": "right",
                "fontsize": 20,
                "frameon": False,
                "bbox_to_anchor": (0.8, 0.90),
            },
            path=Path(plot_dir, f"hlv_corr_{jet_type}.pdf"),
        )
        
        plot_multi_correlations(
            data_list=[np.stack([d[s] for s in feat_spread_vars]).T for d in epic_fm],
            data_labels=[d.label for d in epic_fm],
            col_labels=feat_subvars_dict,
            n_bins=nbins,
            n_kde_points=n_kde_points,
            hist_kwargs=[d.hist_kwargs for d in epic_fm],
            err_kwargs=[d.err_kwargs for d in epic_fm],
            legend_kwargs={
                "loc": "upper right",
                "alignment": "right",
                "fontsize": 20,
                "frameon": False,
                "bbox_to_anchor": (0.8, 0.90),
            },
            path=Path(plot_dir, f"hlv_corr_{jet_type}_2.pdf"),
        )

if __name__ == "__main__":
    main()