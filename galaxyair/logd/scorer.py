"""logD scoring interface used as a post-generation filter.

Per Section 2.3 of the paper:
  "Given the non-linear relationship between logD and optimal membrane
   permeability (typically between 1 and 3), logD is not included in the
   reward function but is used as a filtering condition."
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from galaxyair.logd.model import RTLogDPredictor


class LogDScorer:
    """Thin wrapper around RTLogDPredictor for single-molecule scoring and filtering.

    Parameters
    ----------
    weights_path:
        Path to a saved RTLogDPredictor joblib file.
    logd_min, logd_max:
        Acceptable logD window for CNS penetration.
        Paper guidance: optimal membrane permeability at logD 1–3.
    """

    def __init__(
        self,
        weights_path: str | Path,
        logd_min: float = 1.0,
        logd_max: float = 3.0,
    ) -> None:
        self._predictor = RTLogDPredictor.load(weights_path)
        self._logd_min = logd_min
        self._logd_max = logd_max

    def __call__(self, smiles: str) -> float:
        """Predict logD for a single molecule.

        Returns np.nan for invalid SMILES.
        """
        return self._predictor.predict_single(smiles)

    def is_acceptable(self, smiles: str) -> bool:
        """Return True if the molecule's logD falls within the filter window."""
        logd = self._predictor.predict_single(smiles)
        import math
        if math.isnan(logd):
            return False
        return self._logd_min <= logd <= self._logd_max

    def filter_smiles(
        self, smiles_list: list[str]
    ) -> Tuple[list[str], list[str]]:
        """Split a list of SMILES into (accepted, rejected) by logD.

        Returns
        -------
        accepted:
            SMILES with logD inside [logd_min, logd_max].
        rejected:
            SMILES outside the window or invalid.
        """
        import math
        preds = self._predictor.predict(smiles_list)
        accepted, rejected = [], []
        for smi, logd in zip(smiles_list, preds):
            if not math.isnan(logd) and self._logd_min <= logd <= self._logd_max:
                accepted.append(smi)
            else:
                rejected.append(smi)
        return accepted, rejected
