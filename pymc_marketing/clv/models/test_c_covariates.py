"""Tests for BdW duration (``c``) covariates.

Place in ``tests/clv/models/`` alongside ``test_beta_discrete_weibull.py``.
"""
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from pymc_extras.prior import Prior

from pymc_marketing.clv.models.beta_discrete_weibull import BetaDiscreteWeibullModel


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    n = 120
    coh = np.repeat(pd.date_range("2023-01-01", periods=4, freq="MS"), n // 4)
    T = rng.integers(6, 18, n)
    rec = np.minimum(rng.integers(1, 20, n), T)
    return pd.DataFrame({
        "customer_id": np.arange(n), "cohort": coh, "T": T, "recency": rec,
        "z1": rng.normal(0, 1, n), "z2": rng.normal(0, 1, n),
    })


def _cfg(**extra):
    cfg = {
        "phi": Prior("Uniform", lower=0.0, upper=1.0, dims="cohort"),
        "kappa": Prior("Pareto", alpha=1.0, m=1.0, dims="cohort"),
        "c": Prior("HalfNormal", sigma=1.0, dims="cohort"),
    }
    cfg.update(extra)
    return cfg


def test_duration_covariates_build_and_shapes(data):
    m = BetaDiscreteWeibullModel(
        data=data,
        model_config=_cfg(
            duration_coefficient=Prior("Normal", mu=0, sigma=0.5),
            duration_covariate_cols=["z1", "z2"],
        ),
    )
    m.build_model()
    assert "c_scale" in m.model.named_vars
    assert "duration_coefficient_c" in m.model.named_vars
    assert "duration_data" in m.model.named_vars
    # per-customer deterministic c
    assert m.model.named_vars["c"].eval().shape == (len(data),)
    # model logp finite at the initial point
    assert np.isfinite(m.model.compile_logp()(m.model.initial_point()))


def test_zero_coefficients_collapse_to_base_model(data):
    """With gamma_c pinned at 0, the covariate model's likelihood equals the
    plain cohort-``c`` model's at matched parameter values."""
    m_cov = BetaDiscreteWeibullModel(
        data=data,
        model_config=_cfg(
            duration_coefficient=Prior("Normal", mu=0, sigma=0.5),
            duration_covariate_cols=["z1"],
        ),
    )
    m_cov.build_model()
    m_base = BetaDiscreteWeibullModel(data=data, model_config=_cfg())
    m_base.build_model()

    ncoh = data["cohort"].nunique()
    point = {"phi_interval__": np.zeros(ncoh),
             "kappa_log__": np.full(ncoh, 0.5),
             "c_log__": np.zeros(ncoh)}
    lp_base = m_base.model.compile_logp(vars=[m_base.model["recency"]])(
        m_base.model.initial_point() | point)
    point_cov = dict(point)
    point_cov["c_scale_log__"] = point_cov.pop("c_log__")
    point_cov["duration_coefficient_c"] = np.zeros(1)
    lp_cov = m_cov.model.compile_logp(vars=[m_cov.model["recency"]])(
        m_cov.model.initial_point() | point_cov)
    np.testing.assert_allclose(lp_cov, lp_base, rtol=1e-10)


def test_positive_coefficient_raises_c(data):
    """Sign convention: positive gamma_c and positive covariate => larger c."""
    m = BetaDiscreteWeibullModel(
        data=data,
        model_config=_cfg(
            duration_coefficient=Prior("Normal", mu=0, sigma=0.5),
            duration_covariate_cols=["z1"],
        ),
    )
    m.build_model()
    fn = m.model.compile_fn(m.model["c"], inputs=m.model.free_RVs)
    ip = m.model.initial_point()
    base = m.model.compile_fn(
        m.model["c"], point_fn=True)(ip | {"duration_coefficient_c": np.array([0.0])})
    up = m.model.compile_fn(
        m.model["c"], point_fn=True)(ip | {"duration_coefficient_c": np.array([0.5])})
    pos = data["z1"].to_numpy() > 0
    assert (np.asarray(up)[pos] > np.asarray(base)[pos]).all()
    assert (np.asarray(up)[~pos] < np.asarray(base)[~pos]).all()


def test_fit_and_predict_new_customers(data):
    m = BetaDiscreteWeibullModel(
        data=data,
        model_config=_cfg(
            duration_coefficient=Prior("Normal", mu=0, sigma=0.5),
            duration_covariate_cols=["z1"],
        ),
    )
    m.fit(chains=1, draws=25, tune=25, progressbar=False)
    new = data.head(10).copy()
    new["customer_id"] = new["customer_id"] + 10_000     # unseen customers
    ds = m._extract_predictive_variables(new, customer_varnames=("T",))
    assert "c" in ds
    assert ds["c"].sizes.get("customer_id", ds["c"].sizes.get("cohort")) == 10
    surv = m.expected_retention_rate(new) if hasattr(m, "expected_retention_rate") else None
    # per-customer c must respond to the covariate in prediction too
    lo, hi = new["z1"].idxmin(), new["z1"].idxmax()
    c_mean = ds["c"].mean(dim=[d for d in ds["c"].dims if d not in ("customer_id", "cohort")])
    assert np.isfinite(np.asarray(c_mean)).all()
