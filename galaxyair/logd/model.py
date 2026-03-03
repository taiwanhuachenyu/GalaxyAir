"""
RTlogD: Random Forest logD predictor used as a post-generation filter.

Role in pipeline (Section 2.1 of paper)
-----------------------------------------
logD is NOT part of the RL reward function.  It serves as an output
filtration criterion after molecule generation:
  - Molecules with logD outside the acceptable window are discarded.
  - CNS-focused target range: 0 ≤ logD ≤ 4 (default).

Dataset (Section 2.5 of paper)
---------------------------------
4,200 molecules from ChEMBL with experimental logD values (-1.45 to 4.33).

Features
--------
ECFP-2048 (Morgan radius 2) augmented with RDKit physicochemical descriptors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Featurizer
# ---------------------------------------------------------------------------

_RDKIT_DESCRIPTOR_FNS = [
    ("MolWt",        Descriptors.MolWt),
    ("MolLogP",      Descriptors.MolLogP),
    ("NumHDonors",   Descriptors.NumHDonors),
    ("NumHAcceptors",Descriptors.NumHAcceptors),
    ("TPSA",         Descriptors.TPSA),
    ("NumRotBonds",  rdMolDescriptors.CalcNumRotatableBonds),
    ("NumRings",     rdMolDescriptors.CalcNumRings),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
]


def featurize_smiles(
    smiles_list: List[str],
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    use_rdkit_descriptors: bool = True,
) -> Tuple[np.ndarray, List[bool]]:
    """Convert SMILES to feature matrix.

    Parameters
    ----------
    smiles_list:
        Input SMILES strings.
    morgan_radius:
        Morgan fingerprint radius.
    morgan_n_bits:
        Morgan fingerprint bit length.
    use_rdkit_descriptors:
        Whether to append physicochemical descriptors to the fingerprint.

    Returns
    -------
    (features, valid_mask)
        features: np.ndarray of shape (n_valid, n_features)
        valid_mask: list of bools, True for valid SMILES
    """
    features: List[np.ndarray] = []
    valid_mask: List[bool] = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            continue

        fp = GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_n_bits)
        fp_arr = np.zeros(morgan_n_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, fp_arr)

        if use_rdkit_descriptors:
            desc = np.array(
                [fn(mol) for _, fn in _RDKIT_DESCRIPTOR_FNS], dtype=np.float32
            )
            row = np.concatenate([fp_arr, desc])
        else:
            row = fp_arr

        features.append(row)
        valid_mask.append(True)

    return np.vstack(features) if features else np.empty((0, morgan_n_bits)), valid_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RTLogDPredictor:
    """Random Forest regressor for logD prediction (RTlogD).

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum tree depth (None = unlimited).
    random_state:
        Random seed for reproducibility.
    morgan_radius, morgan_n_bits:
        Morgan fingerprint parameters.
    use_rdkit_descriptors:
        Augment ECFP with physicochemical descriptors.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
        use_rdkit_descriptors: bool = True,
    ) -> None:
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
        self._morgan_radius = morgan_radius
        self._morgan_n_bits = morgan_n_bits
        self._use_rdkit_descriptors = use_rdkit_descriptors
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        smiles_list: List[str],
        logd_values: List[float],
    ) -> "RTLogDPredictor":
        """Train the logD predictor.

        Parameters
        ----------
        smiles_list:
            Training SMILES strings.
        logd_values:
            Experimental logD values.

        Returns
        -------
        self (for chaining)
        """
        features, valid_mask = featurize_smiles(
            smiles_list, self._morgan_radius,
            self._morgan_n_bits, self._use_rdkit_descriptors,
        )
        labels = np.array([v for v, ok in zip(logd_values, valid_mask) if ok])

        if len(features) == 0:
            raise ValueError("No valid molecules found in the training set.")

        logger.info(f"Training RTlogD RF on {len(features)} molecules …")
        self._rf.fit(features, labels)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Return predicted logD values.

        Invalid SMILES get np.nan.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        features, valid_mask = featurize_smiles(
            smiles_list, self._morgan_radius,
            self._morgan_n_bits, self._use_rdkit_descriptors,
        )
        valid_features = features
        predictions = self._rf.predict(valid_features)

        out = np.full(len(smiles_list), np.nan, dtype=np.float64)
        j = 0
        for i, ok in enumerate(valid_mask):
            if ok:
                out[i] = predictions[j]
                j += 1
        return out

    def predict_single(self, smiles: str) -> float:
        """Predict logD for a single molecule. Returns np.nan for invalid input."""
        result = self.predict([smiles])
        return float(result[0])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        smiles_list: List[str],
        logd_values: List[float],
    ) -> dict:
        """Return R², MAE, and RMSE on a labeled set."""
        preds = self.predict(smiles_list)
        valid = ~np.isnan(preds)
        y_true = np.array(logd_values)[valid]
        y_pred = preds[valid]
        return {
            "r2":   float(r2_score(y_true, y_pred)),
            "mae":  float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "n":    int(valid.sum()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "RTLogDPredictor":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_logd_predictor(
    config_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> RTLogDPredictor:
    """Train RTLogDPredictor from a YAML config file."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["paths"]["data"])
    smiles = df[cfg["paths"]["smiles_col"]].tolist()
    logd = df[cfg["paths"]["label_col"]].tolist()

    train_smi, val_smi, train_logd, val_logd = train_test_split(
        smiles, logd, test_size=0.15, random_state=42
    )

    model_cfg = cfg["model"]
    feat_cfg = cfg["featurizer"]
    predictor = RTLogDPredictor(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        random_state=model_cfg["random_state"],
        morgan_radius=feat_cfg["morgan_radius"],
        morgan_n_bits=feat_cfg["morgan_n_bits"],
        use_rdkit_descriptors=feat_cfg["use_rdkit_descriptors"],
    )
    predictor.fit(train_smi, train_logd)

    metrics = predictor.evaluate(val_smi, val_logd)
    logger.info(
        f"Validation — R²: {metrics['r2']:.4f}  "
        f"MAE: {metrics['mae']:.4f}  RMSE: {metrics['rmse']:.4f}"
    )

    save_path = Path(output_dir or cfg["paths"]["weights_path"])
    predictor.save(save_path)
    logger.info(f"RTlogD model saved to {save_path}")

    return predictor
