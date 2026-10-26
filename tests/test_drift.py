"""PSI/KS detector: frozen bins, ε-floor, identical streams do not fire."""

import numpy as np

from eci.eval.drift import frozen_quantile_edges, psi, psi_ks_detector


def test_psi_zero_on_identical_mass() -> None:
    values = np.linspace(1.0, 10.0, 200)
    edges = frozen_quantile_edges(values)
    assert psi(values, values, edges) < 1e-6


def test_psi_ks_detector_triggers_on_shift() -> None:
    rng = np.random.default_rng(7)
    ref = rng.normal(0.0, 1.0, 400)
    shifted = rng.normal(3.0, 1.0, 400)
    edges = frozen_quantile_edges(ref)
    quiet = psi_ks_detector(ref, ref, edges=edges, residual_ref=ref, residual_win=ref)
    loud = psi_ks_detector(ref, shifted, edges=edges, residual_ref=ref, residual_win=shifted)
    assert quiet["triggered"] is False
    assert loud["triggered"] is True
    assert loud["psi"] > quiet["psi"]
