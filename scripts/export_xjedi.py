from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import h5py
import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from src.hydra_utils import reload_original_config

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="export.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(cfg, get_best=cfg.load)

    log.info("Loading best checkpoint")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)

    log.info("Instantiating the data module for the test set")
    datamodule = hydra.utils.instantiate(orig_cfg.datamodule)
    jet_type = datamodule.data_conf.jet_type

    log.info("Creating output directory.")
    outdir = Path("outputs") / jet_type
    outdir.mkdir(exist_ok=True, parents=True)

    log.info("Instantiating the trainer")
    orig_cfg.trainer["enable_progress_bar"] = True
    trainer = hydra.utils.instantiate(orig_cfg.trainer)

    # Cycle through the sampler configurations
    for steps in cfg.sampler_steps:
        for sampler in cfg.sampler_name:
            log.info("Setting up the generation paremeters")
            model.sampler_steps = steps
            model.sampler_name = sampler

            log.info("Running the prediction loop")
            outputs = trainer.predict(model=model, datamodule=datamodule)

            log.info("Combining predictions across dataset")
            keys = list(outputs[0].keys())
            comb_dict = {key: np.vstack([o[key] for o in outputs]) for key in keys}

            log.info("Saving HDF files.")
            with h5py.File(outdir / f"{sampler}_{steps}.h5", mode="w") as file:
                for key in keys:
                    file.create_dataset(key, data=comb_dict[key])


if __name__ == "__main__":
    main()
