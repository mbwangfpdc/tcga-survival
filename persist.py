import pickle
import os
from sksurv.linear_model import CoxPHSurvivalAnalysis

def save_models(model: list[CoxPHSurvivalAnalysis], run_dir: str, model_label: str):
    with open(os.path.join(run_dir, f"{model_label}.pkl"), "wb+") as model_file:
        pickle.dump(model, model_file, protocol=5)

def load_models(path: str) -> list[CoxPHSurvivalAnalysis]:
    with open(path, "rb") as f:
        return pickle.load(f)
