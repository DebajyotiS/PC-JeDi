from typing import Union

import h5py
import jetnet
import numpy as np
import yaml
from jetnet.evaluation import cov_mmd, fpnd, w1efp, w1m, w1p
from scipy.stats import energy_distance, entropy, kstest, wasserstein_distance
from torch import Tensor

from src.physics import numpy_locals_to_mass_and_pt

rng = np.random.default_rng()


def get_output_file_list(path, jet_types, file_names):
    """Returns a list of output file paths based on the given arguments and
    path. Assumes the following folder structure.

    path > jet_type > file_names

    Parameters
    ----------
    jet_types : str
        A comma-separated string of jet types.
    file_names :
        A comma-separated string of file names.
    path : str or Path
        The base path to search for output files.

    Returns
    -------
    list of Path
        A list of output file paths.
    """
    file_paths = []
    for jet_type in jet_types.split(","):
        for f in file_names.split(","):
            fp = path / jet_type
            if "*" in f:
                file_paths += list(fp.glob(f))
            else:
                file_paths.append(fp / f)

    # Filter the paths to ones that actually exist
    file_paths = [f for f in file_paths if f.is_file()]
    if not file_paths:
        print("No matching files found! Exiting!")
    return file_paths


def quantiled_kl_divergence(sample_1: np.ndarray, sample_2: np.ndarray, bins: int = 30):
    "Calculate the kl divergence using quantiles on sample_1 to define the bounds" ""
    bins = np.quantile(sample_1, np.linspace(0.001, 0.999, bins))
    hist_1 = np.histogram(sample_1, bins, normed=True)[0] + 1e-8
    hist_2 = np.histogram(sample_2, bins, normed=True)[0] + 1e-8
    return entropy(hist_1, hist_2)


def get_cov_mmd(
    real_jets: np.ndarray,
    gen_jets: np.ndarray,
    num_eval: int = 100,
    batches: int = 10,
    use_tqdm: bool = False,
) -> np.ndarray:
    """Calculated the coverage and MMD between the real and generated jets,
    using the EMD as a distance metric. Jets are expected to be in (num_jets,
    num_csts, [eta, phi, pt]) format.

    Parameters
    ----------
    real_jets : np.ndarray
        shape: [n_real_jets, n_csts, 3]
    gen_jets : np.ndarray
        shape: [n_gen_jets, n_csts, 3]
    num_eval : int
        The number of jets to evaluate the MMD on.
    use_tqdm : bool, optional
        Whether to use tqdm to show progress, by default True

    Returns
    -------
    Tuple[float, float]
        The coverage and MMD between the real and generated jets.
    """

    coverage, mmd = jetnet.evaluation.cov_mmd(
        real_jets, gen_jets, num_eval, num_batches=batches, use_tqdm=use_tqdm
    )
    return 1 - coverage, mmd


def get_w1efp(
    real_jets: np.ndarray, gen_jets: np.ndarray, num_eval: int = 10000
) -> np.ndarray:
    """Calculated the W1 EFP between the real and generated jets, using the EMD
    as a distance metric. Jets are expected to be in (num_jets, num_csts, [eta,
    phi, pt]) format.

    Parameters
    ----------
    real_jets : np.ndarray
        shape: [n_real_jets, n_csts, 3]
    gen_jets : np.ndarray
        shape: [n_gen_jets, n_csts, 3]
    num_eval : int
        The number of jets to evaluate the MMD on.

    Returns
    -------
    Tuple[float, float]
        The W1 distance between the EFPs and its error, computed between the real and generated jets.
    """

    w1efp, err_w1efp = jetnet.evaluation.w1efp(
        real_jets, gen_jets, num_eval_samples=num_eval, average_over_efps=True
    )
    return w1efp, err_w1efp


def get_w1m(
    real_jets: np.ndarray, gen_jets: np.ndarray, num_eval: int = 10000
) -> np.ndarray:
    """Calculated the W1 distance between the masses of the real and generated
    jets. Jets are expected to be in (num_jets, num_csts, [eta, phi, pt])
    format.

    Parameters
    ----------
    real_jets : np.ndarray
        shape: [n_real_jets, n_csts, 3]
    gen_jets : np.ndarray
        shape: [n_gen_jets, n_csts, 3]
    num_eval : int
        The number of jets to evaluate the MMD on.

    Returns
    -------
    Tuple[float, float]
        The W1 distance between the masses and its error, computed between the real and generated jets.
    """

    w1m, err_w1m = jetnet.evaluation.w1m(real_jets, gen_jets, num_eval_samples=num_eval)
    return w1m, err_w1m


def get_w1p(
    real_jets: np.ndarray,
    gen_jets: np.ndarray,
    real_mask: np.ndarray,
    gen_mask: np.ndarray,
    num_eval: int = 10000,
) -> np.ndarray:
    """Calculated the W1 distance between the particle features of the real and
    generated jets. Jets are expected to be in (num_jets, num_csts, [eta, phi,
    pt]) format.

    Parameters
    ----------
    real_jets : np.ndarray
        shape: [n_real_jets, n_csts, 3]
    gen_jets : np.ndarray
        shape: [n_gen_jets, n_csts, 3]
    real_mask : np.ndarray
        shape: [n_real_jets, n_csts]
    gen_mask : np.ndarray
        shape: [n_gen_jets, n_csts]
    num_eval : int
        The number of jets to evaluate the MMD on.

    Returns
    -------
    Tuple[float, float]
        The W1 distance between the particle features and its error, computed between the real and generated jets.
    """

    w1p, err_w1p = jetnet.evaluation.w1p(
        real_jets, gen_jets, num_eval_samples=num_eval, average_over_features=True
    )
    return w1p, err_w1p


def bootstrapped_marginal_distance(
    sample_1: Union[Tensor, np.ndarray],
    sample_2: Union[Tensor, np.ndarray],
    num_eval_samples: int = 10000,
    num_batches: int = 10,
    metric: str = "w1",
) -> tuple:
    """Get the distance between two distributions using bootstrapping to
    estimate the uncertainties."""
    assert len(sample_1.shape) == 1 and len(sample_2.shape) == 1
    assert metric in ["w1", "w2", "ks", "kldf", "kldr"]

    # Make sure we are using numpy arrays
    if isinstance(sample_1, Tensor):
        sample_1 = sample_1.cpu().detach().numpy()
    if isinstance(sample_2, Tensor):
        sample_2 = sample_2.cpu().detach().numpy()

    # Save a list of the metrics calculated for each bootstrapped batch
    metrics = []
    for i in range(num_batches):

        # Sample with replacement for the batch
        rand1 = rng.choice(len(sample_1), size=num_eval_samples)
        rand2 = rng.choice(len(sample_2), size=num_eval_samples)
        rand_sample1 = sample_1[rand1]
        rand_sample2 = sample_2[rand2]

        # Calculate the metric
        if metric == "w1":
            value = wasserstein_distance(rand_sample1, rand_sample2)
        elif metric == "w2":
            value = energy_distance(rand_sample1, rand_sample2)
        elif metric == "ks":
            value = kstest(rand_sample1, rand_sample2)
        elif metric == "kldf":
            value = quantiled_kl_divergence(rand_sample1, rand_sample2)
        elif metric == "kldr":
            value = quantiled_kl_divergence(rand_sample2, rand_sample1)
        else:
            raise ValueError(f"Unrecognized metric: {metric}")
        metrics.append(value)

    return np.mean(metrics), np.std(metrics)


def dump_metrics(
    gen_path,
    outpath,
    real_path,
    jet_type,
    num_eval_samples,
    num_batches,
    key
):

    # The keyword arguments for the bootstrapping
    bootstrap = {"num_eval_samples": num_eval_samples, "num_batches": num_batches}

    # Start saving everything to a metric dict
    metric_dict = {}

    # Load and clip the generated data
    with h5py.File(gen_path, "r") as f:
        gen_nodes = f["etaphipt_frac"][:]
    gen_nodes[..., 0] = np.clip(gen_nodes[..., 0], -0.5, 0.5)
    gen_nodes[..., 1] = np.clip(gen_nodes[..., 1], -0.5, 0.5)
    gen_nodes[..., 2] = np.clip(gen_nodes[..., 2], 0, 1.0)
    gen_mask = ~np.all(gen_nodes == 0, axis=-1)

    # Load and clip the real data
    with h5py.File(real_path, "r") as f:
        real_nodes = f["etaphipt_frac"][:]
    real_nodes[..., 0] = np.clip(real_nodes[..., 0], -0.5, 0.5)
    real_nodes[..., 1] = np.clip(real_nodes[..., 1], -0.5, 0.5)
    real_nodes[..., 2] = np.clip(real_nodes[..., 2], 0, 1.0)
    real_mask = ~np.all(real_nodes == 0, axis=-1)

    # The FPND metric, not always valid
    if jet_type in ["g", "t", "q"]:
        if gen_nodes.shape[-2] > 30:
            sort_idx = np.argsort(gen_nodes[..., 2], axis=-1)[..., None]
            top_30 = np.take_along_axis(gen_nodes, sort_idx, axis=1)
            top_30 = top_30[:, -30:]
            fpnd_val = fpnd(top_30, jet_type=jet_type)
        else:
            fpnd_val = fpnd(gen_nodes, jet_type=jet_type)
    else:
        fpnd_val = 0
    print(f"fpnd:  {fpnd_val:4.3E}")
    metric_dict["fpnd"] = fpnd_val

    # The standard JetNet Metrics
    w1m_val, w1m_err = w1m(real_nodes, gen_nodes, **bootstrap)
    print(f"w1m:   {w1m_val:4.3E} +- {w1m_err:5.4E}")
    metric_dict.update({"w1m": w1m_val, "w1m_err": w1m_err})

    w1p_val, w1p_err = w1p(real_nodes, gen_nodes, **bootstrap)
    w1p_val = w1p_val.mean()
    w1p_err = w1p_err.mean()
    print(f"w1p:   {w1p_val:4.3E} +- {w1p_err:5.4E}")
    metric_dict.update({"w1p": w1p_val, "w1p_err": w1p_err})

    w1efp_val, w1efp_err = w1efp(real_nodes, gen_nodes, **bootstrap, efp_jobs=1)
    w1efp_val = w1efp_val.mean()
    w1efp_err = w1efp_err.mean()
    print(f"w1efp: {w1efp_val:4.3E} +- {w1efp_err:5.4E}")
    metric_dict.update({"w1efp": w1efp_val, "w1efp_err": w1efp_err})

    cov, mmd = cov_mmd(real_nodes, gen_nodes, num_eval_samples=400)
    print(f"cov:   {cov:4.3E}")
    print(f"mmd:   {mmd:4.3E}")
    metric_dict.update({"cov": cov, "mmd": mmd})

    # The mass and pt obedience metrics
    if np.all(gen_mask == real_mask):
        with h5py.File(gen_path, "r") as f:
            gen_nodes = f["etaphipt_frac"][:]
        with h5py.File(real_path, "r") as f:
            real_nodes = f["etaphipt_frac"][:]
        gen_pc_jet = numpy_locals_to_mass_and_pt(gen_nodes, gen_mask)
        real_pc_jet = numpy_locals_to_mass_and_pt(real_nodes, real_mask)
        pt_mae = np.mean(np.abs(real_pc_jet[..., 0] - gen_pc_jet[..., 0]))
        mass_mae = np.mean(np.abs(real_pc_jet[..., 1] - gen_pc_jet[..., 1]))
    else:
        pt_mae = 0
        mass_mae = 0
    print(f"pt_mae   {pt_mae:4.3E}")
    print(f"m_mae:   {mass_mae:4.3E}")
    metric_dict.update({"pt_mae": pt_mae, "mass_mae": mass_mae})

    # Our full suite of substructure metrics
    gen_subfile = gen_path.parent / gen_path.stem.replace("csts", f"{key}.h5")
    real_subfile = real_path.parent / real_path.stem.replace("csts", f"{key}.h5")
    try:
        with h5py.File(gen_subfile, "r") as f:
            with h5py.File(real_subfile, "r") as rf:
                for k in f.keys():
                    gen_sub = f[k][:].flatten()
                    real_sub = rf[k][:].flatten()
                    # Drop the nan values
                    gen_sub = gen_sub[~np.isnan(gen_sub)]
                    real_sub = real_sub[~np.isnan(real_sub)]
                    for metric in ["w1"]:
                        metric_val, metric_std = bootstrapped_marginal_distance(
                            gen_sub, real_sub, metric=metric, **bootstrap
                        )
                        print(f"{metric}_{k}: {metric_val:4.3E} +- {metric_std:5.4E}")
                        metric_dict[f"{metric}_{k}"] = metric_val
                        metric_dict[f"{metric}_{k}_err"] = metric_std
    except FileNotFoundError:
        print(f"Could not find {gen_subfile}. Skipping substructure metrics.")
        
    # Save the entire dictionary into a yaml file
    metric_dict = {k: float(v) for k, v in metric_dict.items()}
    with open(outpath, "w") as f:
        yaml.dump(metric_dict, f)
    print("")
