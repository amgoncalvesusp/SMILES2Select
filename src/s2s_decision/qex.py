"""Independent QEX equation implementation, not IMSBIO QEX Suite.

Mochizuki et al., doi:10.1007/s11030-018-9842-3. Fits eight active-only
histograms with positive asymmetric double sigmoids and the published exhaustive
entropy weighting. Histogram choices and positive parameterization are declared
in the model; the publication does not specify them. Scores are not probabilities.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED
from scipy.optimize import brentq, least_squares
from scipy.special import expit

PROPERTY_NAMES = QED.QEDproperties._fields
_COUNTS = (2, 3, 5, 6, 7)
_METHOD = 'qex_published_equations_independent_v1'
_TINY = np.finfo(np.float64).tiny


class QEXFitError(ValueError):
    """Published-equation fit unavailable; caller must report, not replace it."""


@dataclass(frozen=True)
class QEXCurve:
    name: str
    location: float
    scale: float
    parameters: tuple[float, ...]
    normalizer: float
    histogram_bins: int
    rmse: float
    converged_starts: int


@dataclass(frozen=True)
class QEXModel:
    curves: tuple[QEXCurve, ...]
    weights: tuple[float, ...]
    training_count: int
    training_sha256: str
    rdkit_version: str

    def __post_init__(self):
        if not isinstance(self.curves, tuple) or not isinstance(self.weights, tuple):
            raise ValueError('QEX model containers must be immutable tuples')
        if tuple(c.name for c in self.curves) != PROPERTY_NAMES:
            raise ValueError('QEX needs all eight properties in QED order')
        w = np.asarray(self.weights, dtype=float)
        if w.shape != (8,) or not np.isfinite(w).all() or (w < 0).any() or (w > 1).any() or w.sum() <= 0:
            raise ValueError('QEX weights must be finite in [0,1], with positive sum')
        if type(self.training_count) is not int or self.training_count < 8:
            raise ValueError('QEX training_count must be an integer >= 8')
        if len(self.training_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.training_sha256):
            raise ValueError('QEX training_sha256 is invalid')
        if not isinstance(self.rdkit_version, str) or not self.rdkit_version:
            raise ValueError('QEX RDKit version is required')
        for curve in self.curves:
            _validate_curve(curve)

    def to_dict(self) -> dict:
        result = asdict(self)
        return {**result, 'method': _METHOD, 'weights': list(self.weights),
                'properties': list(PROPERTY_NAMES), 'weight_candidates': 390624,
                'weight_top_count': 1000, 'histogram': 'unit_count_or_sqrt_continuous_v1',
                'optimizer': 'positive_softplus_LM_three_starts', 'score_floor': _TINY}

    @classmethod
    def from_dict(cls, value: Mapping) -> QEXModel:
        if not isinstance(value, Mapping) or value.get('method') != _METHOD:
            raise ValueError('Unrecognized QEX model method')
        try:
            if tuple(value['properties']) != PROPERTY_NAMES:
                raise ValueError('QEX property order mismatch')
            constants = {'weight_candidates': 390624, 'weight_top_count': 1000,
                         'histogram': 'unit_count_or_sqrt_continuous_v1',
                         'optimizer': 'positive_softplus_LM_three_starts', 'score_floor': _TINY}
            if any(value.get(key) != expected for key, expected in constants.items()):
                raise ValueError('QEX method settings mismatch')
            curves = tuple(QEXCurve(**{**c, 'parameters': tuple(c['parameters'])}) for c in value['curves'])
            return cls(curves, tuple(value['weights']), value['training_count'],
                       value['training_sha256'], value['rdkit_version'])
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('Invalid QEX model structure') from exc


def _validate_curve(curve: QEXCurve) -> None:
    if not isinstance(curve.parameters, tuple):
        raise ValueError('QEX parameters must be immutable tuples')
    p = np.asarray(curve.parameters, dtype=float)
    scalars = [curve.location, curve.scale, curve.normalizer, curve.rmse]
    if p.shape != (6,) or not np.isfinite(p).all() or not np.isfinite(scalars).all():
        raise ValueError('QEX curve parameters must be finite')
    if np.max(np.abs(p)) >= 1e8:
        raise ValueError('QEX curve parameters exceed the fitted admissible range')
    if p[0] < 0 or min(p[1], p[3], p[4], p[5], curve.scale, curve.normalizer) <= 0:
        raise ValueError('QEX curve widths, amplitude, scale and normalizer must be positive')
    if curve.rmse < 0 or type(curve.histogram_bins) is not int or curve.histogram_bins < 6:
        raise ValueError('Invalid QEX fit diagnostics')
    if type(curve.converged_starts) is not int or not 1 <= curve.converged_starts <= 3:
        raise ValueError('QEX fit must have a converged start')
    if not np.isclose(_normalizer(p), curve.normalizer, rtol=1e-8, atol=1e-12):
        raise ValueError('QEX normalizer does not match curve maximum')


def qed_properties(smiles: Sequence[str]) -> np.ndarray:
    """Calculate the actual QED property definitions; do not substitute PAINS counts."""
    if isinstance(smiles, (str, bytes)):
        raise ValueError('Expected a sequence of individual SMILES strings')
    rows = []
    for index, value in enumerate(smiles):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Nonempty SMILES required at index {index}')
        mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f'Invalid SMILES at index {index}')
        rows.append(tuple(QED.properties(mol)))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def _properties(values, *, training: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 8 or not np.isfinite(result).all():
        raise ValueError('QEX requires a finite (n,8) property matrix in QED order')
    if training and len(result) < 8:
        raise ValueError('QEX requires at least eight active training molecules')
    if (result[:, (0, 2, 3, 4, 5, 6, 7)] < 0).any():
        raise ValueError('QEX physical properties other than ALOGP must be nonnegative')
    if not np.equal(result[:, _COUNTS], np.floor(result[:, _COUNTS])).all():
        raise ValueError('QEX HBA, HBD, ROTB, AROM and ALERTS must be integer counts')
    return result


def _ads(x: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
    a, b, c, d, e, f = parameters
    # Opposite-sign second sigmoid avoids catastrophic 1-sigmoid cancellation.
    with np.errstate(over='ignore'):
        return a + b * expit((x - c + d / 2) / e) * expit(-(x - c - d / 2) / f)


def _normalizer(parameters: Sequence[float]) -> float:
    _, _, c, d, e, f = parameters
    radius = d / 2 + 80 * max(e, f)
    def derivative(x):
        return expit(-(x - c + d / 2) / e) / e - expit((x - c - d / 2) / f) / f
    mode = brentq(derivative, c - radius, c + radius)
    return float(_ads(np.asarray(mode), parameters))


def _unpack(raw: np.ndarray) -> tuple[float, ...]:
    p = np.logaddexp(0, raw)
    return (float(p[0]), float(p[1]), float(raw[2]),
            float(p[3] + 1e-6), float(p[4] + 1e-6), float(p[5] + 1e-6))


def _histogram(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    lo, hi = float(values.min()), float(values.max())
    if hi == lo:
        raise QEXFitError(f'QEX unavailable: {name} is constant in active training data')
    location, scale = float(np.mean(values)), float(np.std(values))
    if name in tuple(PROPERTY_NAMES[i] for i in _COUNTS):
        if hi - lo > 10000:
            raise QEXFitError(f'QEX unavailable: {name} count range exceeds 10000')
        edges = np.arange(lo - 3.5, hi + 4.5, 1.0)
    else:
        bins = max(12, min(40, int(np.ceil(np.sqrt(len(values))))))
        edges = np.linspace(lo - (hi-lo) * .1, hi + (hi-lo) * .1, bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    x = ((edges[:-1] + edges[1:]) / 2 - location) / scale
    return x, counts / counts.max(), location, scale


def _fit_curve(values: np.ndarray, name: str) -> QEXCurve:
    x, y, location, scale = _histogram(values, name)
    # Publication leaves starts/binning unspecified. Choices are fixed before evaluation.
    starts = ((-8, 1, 0, 1, -1, -1), (-8, 2, 0, 2, 0, 0), (-5, 1, 0, 0, -2, 0))
    candidates = []
    failures = []
    for start in starts:
        try:
            result = least_squares(lambda r: _ads(x, _unpack(r)) - y, start,
                                   method='lm', max_nfev=4000)
        except (ValueError, FloatingPointError) as error:
            failures.append(str(error))
            continue
        params = _unpack(result.x)
        if result.success and np.isfinite(params).all() and max(abs(v) for v in params) < 1e8:
            if params[1] > 0 and np.isfinite(result.cost):
                candidates.append((float(result.cost), params))
    if not candidates:
        raise QEXFitError(f'QEX unavailable: no converged positive LM fit for {name}; {failures}')
    cost, params = min(candidates, key=lambda item: item[0])
    return QEXCurve(name, location, scale, params, _normalizer(params), len(x),
                    float(np.sqrt(2 * cost / len(x))), len(candidates))


def _log_desirability(curves: tuple[QEXCurve, ...], values: np.ndarray) -> np.ndarray:
    rows = [_ads((values[:, i] - curve.location) / curve.scale, curve.parameters)
            / curve.normalizer for i, curve in enumerate(curves)]
    return np.log(np.clip(np.column_stack(rows), _TINY, 1.0))


def _entropy_weights(log_desirability: np.ndarray) -> tuple[float, ...]:
    grid = np.asarray(list(product((0, .25, .5, .75, 1), repeat=8)), dtype=float)[1:]
    # ponytail: exhaustive published grid, batch only memory; no sampled weight shortcut.
    unique, counts = np.unique(log_desirability, axis=0, return_counts=True)
    entropy = []
    for start in range(0, len(grid), 1024):
        weights = grid[start:start + 1024]
        log_scores = unique @ (weights / weights.sum(axis=1, keepdims=True)).T
        entropy.append(-np.sum(counts[:, None] * np.exp(log_scores) * log_scores, axis=0) / np.log(2))
    values = np.concatenate(entropy)
    top = np.lexsort((np.arange(len(grid)), -values))[:1000]
    return tuple(float(w) for w in grid[top].mean(axis=0))


def fit_qex(active_properties: np.ndarray) -> QEXModel:
    """Fit active TRAIN data only; caller owns frozen split and endpoint provenance."""
    values = _properties(active_properties, training=True)
    curves = tuple(_fit_curve(values[:, i], name) for i, name in enumerate(PROPERTY_NAMES))
    weights = _entropy_weights(_log_desirability(curves, values))
    digest = hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()
    return QEXModel(curves, weights, len(values), digest, rdBase.rdkitVersion)


def predict_qex(model: QEXModel, properties: np.ndarray) -> np.ndarray:
    """Score with sealed curves/weights; prediction batches never alter the model."""
    values = _properties(properties)
    w = np.asarray(model.weights)
    return np.exp(_log_desirability(model.curves, values) @ (w / w.sum()))
