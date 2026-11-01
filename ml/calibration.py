"""Calibration wrapper used by the model bundle.

Kept in its own module (not in ``train.py``) so that a model trained with
``python -m ml.train`` pickles this class as ``ml.calibration`` rather than
``__main__``; otherwise the bundle could not be loaded by any other program.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


class RareClassCalibratedCV(CalibratedClassifierCV):
    """``CalibratedClassifierCV`` whose internal folds never outnumber the rarest class.

    Calibration fits the model on internal stratified folds, and stratified
    k-fold needs at least k rows of every class. Inside an out-of-fold loop on
    the training months the rarest class can have only a handful of rows, so the
    number of internal folds is lowered to that count for that fit (5 otherwise).
    """

    def fit(self, X, y, **fit_params):
        configured = self.cv
        rarest = int(pd.Series(np.asarray(y)).value_counts().min())
        self.cv = StratifiedKFold(min(configured.get_n_splits(), rarest), shuffle=True,
                                  random_state=configured.random_state)
        try:
            return super().fit(X, y, **fit_params)
        finally:
            self.cv = configured   # keep the configured parameter, so clones behave the same
