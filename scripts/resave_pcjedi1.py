import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import numpy as np
import h5py
from pathlib import Path
from jetnet.datasets import JetNet

from src.numpy_utils import undo_log_squash


real_nodes, real_jets = JetNet.getData(
        jet_type="t",
        data_dir="/srv/beegfs/scratch/groups/rodem/datasets/jetnet/",
        split="test",
        split_fraction=[0.7, 0.0, 0.3],
        particle_features=["etarel", "phirel", "ptrel"],
        jet_features=["pt", "mass"],
    )


pcjedi_path = Path("/srv/beegfs/scratch/groups/rodem/jet_diffusion/exported/paper_models/jetnet_62737757_0")
file_names = ["ddim_200.h5", "em_200.h5"]

pt = real_jets[:,0]
file_h5 = list(pcjedi_path.glob("*200.h5"))
print(file_h5)
for file in file_h5:
    with h5py.File(file, "r") as f:
        nodes = f["generated"][:]
        etaphipt_frac = nodes.copy()
        etaphipt = nodes.copy()
        etaphipt[:,:,-1] = undo_log_squash(etaphipt[:,:,-1])
        etaphipt_frac[:,:,-1] = undo_log_squash(etaphipt_frac[:,:,-1])/pt[:,None]
        with h5py.File(file.stem+"_csts.h5", "w") as hf:
            hf.create_dataset("etaphipt", data=etaphipt)
            hf.create_dataset("etaphipt_frac", data=etaphipt_frac)
        print("done")
