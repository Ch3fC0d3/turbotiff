"""Deterministic review ranking with uncertainty, topology, and novelty signals."""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class ReviewWeights:
    uncertainty: float = 0.35
    disagreement: float = 0.20
    uncertain_wrap: float = 0.15
    topology_warning: float = 0.10
    novelty: float = 0.20


def novelty_score(feature, reviewed_features) -> float:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    reviewed = np.asarray(reviewed_features, dtype=np.float32)
    if reviewed.size == 0:
        return 1.0
    if reviewed.ndim != 2 or reviewed.shape[1] != feature.size:
        raise ValueError("Reviewed embeddings must have shape [N, D]")
    distance = float(np.min(np.linalg.norm(reviewed - feature[None], axis=1)))
    return float(1.0 - np.exp(-distance))


def rank_case(signals: dict, feature=None, reviewed_features=(), weights: ReviewWeights | None = None) -> dict:
    weights = weights or ReviewWeights()
    novelty = novelty_score(feature, reviewed_features) if feature is not None else float(signals.get("novelty", 0.0))
    components = {
        "uncertainty": float(np.clip(signals.get("uncertainty", 0), 0, 1)),
        "disagreement": float(np.clip(signals.get("disagreement", 0), 0, 1)),
        "uncertain_wrap": float(np.clip(signals.get("uncertain_wrap", 0), 0, 1)),
        "topology_warning": float(np.clip(signals.get("topology_warning", 0), 0, 1)),
        "novelty": novelty,
    }
    score = sum(components[key] * getattr(weights, key) for key in components)
    return {"review_priority": float(score), "components": components, "weights": asdict(weights),
            "review_units": list(signals.get("review_units", []))}


def mixed_epoch_indices(source_sizes: dict[str, int], ratios: dict[str, float], epoch_size: int, seed: int) -> list[tuple[str, int]]:
    if epoch_size < 1 or any(source_sizes.get(name, 0) < 1 for name, ratio in ratios.items() if ratio > 0):
        raise ValueError("Every enabled source needs samples and epoch_size must be positive")
    rng = np.random.default_rng(seed)
    names = sorted(name for name, ratio in ratios.items() if ratio > 0)
    probabilities = np.asarray([ratios[name] for name in names], dtype=float); probabilities /= probabilities.sum()
    chosen = rng.choice(names, size=epoch_size, p=probabilities)
    return [(name, int(rng.integers(source_sizes[name]))) for name in chosen]
