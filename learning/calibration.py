"""Dependency-free confidence calibration metrics and binned calibrator."""

import numpy as np


class BinnedCalibrator:
    def __init__(self, bins: int = 10): self.bins = int(bins); self.accuracy = None
    def fit(self, confidence, correct):
        confidence = np.clip(np.asarray(confidence, dtype=float), 0, 1)
        correct = np.asarray(correct, dtype=float)
        indexes = np.minimum((confidence * self.bins).astype(int), self.bins - 1)
        self.accuracy = np.array([correct[indexes == i].mean() if np.any(indexes == i) else (i + .5) / self.bins for i in range(self.bins)])
        return self
    def transform(self, confidence):
        if self.accuracy is None: raise RuntimeError("Calibrator is not fitted")
        confidence = np.clip(np.asarray(confidence, dtype=float), 0, 1)
        return self.accuracy[np.minimum((confidence * self.bins).astype(int), self.bins - 1)]
    def to_dict(self): return {"method": "binned", "bins": self.bins, "accuracy": self.accuracy.tolist()}


def calibration_report(confidence, errors, tolerance: float, bins: int = 10) -> dict:
    confidence = np.asarray(confidence, dtype=float); correct = np.asarray(errors, dtype=float) <= tolerance
    calibrated = BinnedCalibrator(bins).fit(confidence, correct)
    predicted = calibrated.transform(confidence)
    return {"expected_calibration_error": float(np.mean(np.abs(predicted - correct))), "calibrator": calibrated.to_dict()}
