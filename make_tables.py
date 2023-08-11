import pandas as pd
import numpy as np
from pathlib import Path

import yaml

num_const = 30
cond=True

cond_30_paths = {
    "EPiC-JeDi":"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/final_cedric_changes/2023-07-16_09-58-26-914027/",
    "EPiC-FM":"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/epic_fm/30/"
}

uncond_30_paths = {
    "EPiC-JeDi":"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/final_cedric_changes/2023-07-16_09-58-26-913806/",
    "EPiC-FM":"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/epic_fm/30_uncond/"
}

cond_150_paths = {
    "EPiC-JeDi":"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/final_cedric_changes/2023-07-16_09-58-26-904758/",
    "EPiC-FM":"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/epic_fm/150/"
}

uncond_150_paths = {
    "EPiC-JeDi":"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/final_cedric_changes/2023-07-16_09-58-26-905101/",
    "EPiC-FM":"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/epic_fm/150_uncond/"
}

score_files = ["em_200","midpoint_100", "euler_200"]
score_files_dict = {
    "em_200":"EM",
    "midpoint_100":"Midpoint",
    "euler_200":"Euler"
}

scores = ["fpnd", "w1m", "w1p", "w1efp", "w1_tau21", "w1_tau32", "w1_d2"]
err_scores = [f"{score}_err" for score in scores[1:]]

scores_dict = {
    "fpnd":"FPND",
    "w1m":r"$\mathrm{W}_1^{M} (\times 10{^-4})$",
    "w1p":r"$\mathrm{W}_1^{P} (\times 10{^-4})$",
    "w1efp":r"$\mathrm{W}_1^{EFP} (\times 10{^-5})$",
    "w1_tau21":r"$\mathrm{W}_1^{\tau_{21}} (\times 10{^-3})$",
    "w1_tau32":r"$\mathrm{W}_1^{\tau_{32}} (\times 10{^-3})$",
    "w1_d2":r"$\mathrm{W}_1^{D_2} (\times 10{^-3})$",
}


scores_multiplier = {
    "fpnd":1,
    "w1m":1E4,
    "w1p":1E4,
    "w1efp":1E5,
    "w1_tau21":1E3,
    "w1_tau32":1E3,
    "w1_d2":1E3,
}

err_multiplier = {
    "w1m_err":1E4,
    "w1p_err":1E4,
    "w1efp_err":1E5,
    "w1_tau21_err":1E3,
    "w1_tau32_err":1E3,
    "w1_d2_err":1E3,
}

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def get_scores(path, score_file):
    if isinstance(path, str):
        path = Path(path)
    with open(path/f"outputs/t/{score_file}_scores.yaml") as f:
        scores = yaml.load(f, Loader=yaml.FullLoader)
    return scores

def filter_scores(score_file, scores, err_scores):
    score = {k:v for k,v in score_file.items() if k in scores}
    # Include the corresponding errors in the same dict, if they exist
    errors = {k:v for k,v in score_file.items() if k in err_scores}
    return merge_two_dicts(score, errors)
print("here")


#Collate the scores for epic jedi and epic fm for the 30 cond.
# The top level keys are EPiC-JeDi and EPiC-FM,
# the second level keys are the score_files
# The columns are the scores and the errors

def stream_to_latex(cond_30_paths, name):
    cond_30_scores = {k:{} for k in cond_30_paths.keys()}
    for model, path in cond_30_paths.items():
        score_file = {}
        for file in score_files:
            try:
                score_file = merge_two_dicts(score_file, get_scores(path, file))
                print(f"Got scores for {model} {file}")
                cond_30_scores[model][score_files_dict[file]] = filter_scores(score_file, scores, err_scores)
            except FileNotFoundError:
                print(f"File {file} not found for {model}")
    
    #Make a multiindex dataframe for the 30 cond scores - the top level index is the model, the second level index is the score file, the columns are the scores and the errors
    #Drop the columns that are all NaN
    
    cond_30_scores_df = pd.concat({k:pd.DataFrame(v).T for k,v in cond_30_scores.items()}, axis=0, names=["Model", "Sampler"]).dropna(axis=1, how="all")
    
    # Arrange the columns as FPND, W1M +- err, W1P +- err, W1EFP +- err, W1Tau21 +- err, W1Tau32 +- err
    cond_30_scores_df = cond_30_scores_df[["fpnd", "w1m", "w1m_err", "w1p", "w1p_err", "w1efp", "w1efp_err", "w1_tau21", "w1_tau21_err", "w1_tau32", "w1_tau32_err", "w1_d2", "w1_d2_err"]]
    # Multiply the scores by the appropriate factor
    cond_30_scores_df = cond_30_scores_df.apply(lambda x: x*scores_multiplier[x.name] if x.name in scores_multiplier.keys() else x)
    #Multiply the errors by the appropriate factor
    cond_30_scores_df = cond_30_scores_df.apply(lambda x: x*err_multiplier[x.name] if x.name in err_multiplier.keys() else x)
    
    
    # Make new columns that incomporate the errrors into the score with a $\pm$ notation for scores that have an error
    cond_30_scores_df["fpnd"] = cond_30_scores_df.apply(lambda x: f"${x['fpnd']:.2f}$", axis=1)
    cond_30_scores_df["w1m_pm"] = cond_30_scores_df.apply(lambda x: f"${x['w1m']:.2f} \pm {x['w1m_err']:.2f}$", axis=1)
    cond_30_scores_df["w1p_pm"] = cond_30_scores_df.apply(lambda x: f"${x['w1p']:.2f} \pm {x['w1p_err']:.2f}$", axis=1)
    cond_30_scores_df["w1efp_pm"] = cond_30_scores_df.apply(lambda x: f"${x['w1efp']:.2f} \pm {x['w1efp_err']:.2f}$", axis=1)
    cond_30_scores_df["w1_tau21_pm"] = cond_30_scores_df.apply(lambda x: f"${x['w1_tau21']:.2f} \pm {x['w1_tau21_err']:.2f}$", axis=1)
    cond_30_scores_df["w1_tau32_pm"] = cond_30_scores_df.apply(lambda x: f"${x['w1_tau32']:.2f} \pm {x['w1_tau32_err']:.2f}$", axis=1)
    cond_30_scores_df["w1_d2_pm"] = cond_30_scores_df.apply(lambda x: f"${x['w1_d2']:.2f} \pm {x['w1_d2_err']:.2f}$", axis=1)
    
    
    # Drop the columns for errors and individual scores
    cond_30_scores_df = cond_30_scores_df.drop(columns=["w1m_err", "w1p_err", "w1efp_err", "w1_tau21_err", "w1_tau32_err", "w1m", "w1p", "w1efp", "w1_tau21", "w1_tau32", "w1_d2", "w1_d2_err"])
    
    err_dict_rename = {
        "fpnd":r"\textbf{FPND}",
        "w1m_pm":r"$\mathbf{\mathrm{W}_1^{M} (\times 10{^{-4}})}$",
        "w1p_pm":r"$\mathbf{\mathrm{W}_1^{P} (\times 10{^{-4}})}$",
        "w1efp_pm":r"$\mathbf{\mathrm{W}_1^{EFP} (\times 10{^{-5}})}$",
        "w1_tau21_pm":r"$\mathbf{\mathrm{W}_1^{\tau_{21}} (\times 10{^{-3}})}$",
        "w1_tau32_pm":r"$\mathbf{\mathrm{W}_1^{\tau_{32}} (\times 10{^{-3}})}$",
        "w1_d2_pm":r"$\mathbf{\mathrm{W}_1^{D_2} (\times 10{^{-3}})}$",
    }
    #Rename the columns to the latex names
    cond_30_scores_df = cond_30_scores_df.rename(columns=err_dict_rename)
    
    # Each score column has a string value either of the form "$score$" or "$score \pm err$".
    # We want to get the index of the minimum score for and then add \textbf{} to the latex string for that score
    
    def get_min_score_index(row):
        min_score = np.inf
        min_score_index = None
        for i, score in enumerate(row):
            # see if score is of the form "$score \pm err$" or "$score$"
            if "\pm" in score:
                score = score.split(" \pm ")[0].strip("$")
            else:
                score = score.strip("$")
            score = float(score)
            if score < min_score:
                min_score = score
                min_score_index = i
        return min_score_index
    
    def add_textbf(row):
        min_score_index = get_min_score_index(row)
        if min_score_index is not None:
            # add \mathbf{} to the latex string for that score. the \mathbf should be inside the $$
            # replace the first $ with \mathbf{$ and the last $ with $}
            row[min_score_index] = "$\mathbf{"+row[min_score_index].strip("$")+"}$"
        return row
    
    cond_30_scores_df = cond_30_scores_df.apply(add_textbf, axis=0)
    with open(name, "w") as f:
        cond_30_scores_df.style.to_latex(f, column_format="ll"+"r"*len(err_dict_rename), hrules=True, position="ht")

stream_to_latex(cond_30_paths, name="cond_30_scores.tex")
stream_to_latex(uncond_30_paths, name="uncond_30_scores.tex")
stream_to_latex(cond_150_paths, name="cond_150_scores.tex")
stream_to_latex(uncond_150_paths, name="uncond_150_scores.tex")
