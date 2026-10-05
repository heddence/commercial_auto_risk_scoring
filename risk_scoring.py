"""Shared preparation, model construction, and saved-model scoring.

The notebook owns model comparison and evaluation. This module contains only
the operations that must behave identically during training and scoring.
"""
from pathlib import Path
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

SEED = 42
CATEGORICAL = ["coverage_type", "business_type", "state", "payment_frequency"]
NUMERIC = [
    "vehicle_count", "vehicle_avg_age", "driver_count", "driver_avg_age",
    "years_in_business", "prior_year_mileage_000", "prior_apd_claim_count",
    "prior_al_claim_count", "prior_loss_amount", "deductible", "coverage_limit_000",
    "annual_premium", "risk_score_external", "num_heavy_vehicles",
]
FEATURES = CATEGORICAL + NUMERIC
COUNTS = ["vehicle_count", "driver_count", "num_heavy_vehicles",
          "prior_apd_claim_count", "prior_al_claim_count"]
DERIVED = ["prior_claims_same_coverage", "log_prior_loss", "log_mileage_per_vehicle",
           "heavy_vehicle_share", "drivers_per_vehicle", "al_vehicle_count",
           "al_external_risk"]
DATES = ["snapshot_date", "policy_effective_date", "policy_expiration_date"]


def prepare_data(raw, *, training=False):
    """Normalize labels, validate identity/dates, and mark invalid inputs unknown.

    Returns the prepared rows, an audit report, and excluded training rows.
    """
    required = FEATURES + ["policy_id", "insured_id"] + DATES
    if training:
        required += ["claim_count"]
    
    if raw.empty:
        raise ValueError("The input contains no policies")

    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # These rules are fixed from the field definitions, not learned from test data.
    data = raw.copy()
    for col in CATEGORICAL:
        values = data[col]

        if not values.dropna().map(lambda x: isinstance(x, str)).all():
            raise ValueError(f"{col} must contain text or missing values")

        values = values.astype("string").str.strip().replace("", pd.NA)
        if col == "business_type":
            values = values.str.lower()
        elif col in ["coverage_type", "state"]:
            values = values.str.upper()
        else:
            values = values.str.title()
        data[col] = values.astype(object).where(values.notna(), np.nan)
        
    if not data.coverage_type.isin(["APD", "AL"]).all():
        raise ValueError("coverage_type must be APD or AL")

    before = len(data)
    if training:
        # Normalize first: case differences alone should not count as new policies
        data = data.drop_duplicates().copy()
    
    for col in ["policy_id", "insured_id"]:
        if data[col].isna().any() or not data[col].map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all():
            raise ValueError(f"{col} must contain nonempty text")

    if data.policy_id.duplicated().any():
        raise ValueError("Duplicate policy_id values remain; resolve them before proceeding")

    for col in DATES:
        data[col] = pd.to_datetime(data[col], format="%Y-%m-%d", errors="raise")
        if data[col].isna().any():
            raise ValueError(f"{col} cannot be missing")
    
    invalid_period = data['policy_expiration_date'].le(data['policy_effective_date'])
    rejected = data.loc[invalid_period].copy()
    if invalid_period.any() and not training:
        raise ValueError("Scoring contains a policy whose expiration is not after inception")

    data = data.loc[~invalid_period].reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid policy periods remain")

    if training:
        # The outcome is a target. It is never included in the model's FEATURES.
        counts = pd.to_numeric(data.claim_count, errors="raise")
        invalid = counts.isna() | ~np.isfinite(counts) | counts.lt(0) | counts.mod(1).ne(0)
        if invalid.any():
            raise ValueError("claim_count must contain observed, nonnegative whole numbers")
        data["claim_count"] = counts.astype(int)
        data["had_claim"] = counts.gt(0).astype(int)
    
    # Invalid predictor values become unknown; the model handles missing inputs
    issues = {}
    for col in NUMERIC:
        parsed = pd.to_numeric(data[col], errors="coerce").astype(float)
        invalid = ~np.isfinite(parsed) & data[col].notna()
        invalid |= parsed.lt(0)
        if col in COUNTS:
            invalid |= parsed.notna() & parsed.mod(1).ne(0)
        if col == "risk_score_external":
            invalid |= parsed.notna() & ~parsed.between(0, 100)
        if col == "deductible":
            invalid |= parsed.notna() & ~parsed.isin([250, 500, 1000, 2500, 5000])
        issues[col] = int(invalid.sum())
        data[col] = parsed.mask(invalid)
    fleet_conflict = data['num_heavy_vehicles'].gt(data['vehicle_count'])
    data.loc[fleet_conflict, ["vehicle_count", "num_heavy_vehicles"]] = np.nan

    report = {
        "input_rows": len(raw),
        "duplicate_copies_removed": before - len(data) - len(rejected),
        "invalid_periods_excluded": len(rejected),
        "retained_rows": len(data),
        "invalid_inputs_marked_missing": {k: v for k, v in issues.items() if v},
        "inconsistent_fleet_pairs_marked_missing": int(fleet_conflict.sum()),
        "missing_predictors": data[FEATURES].isna().sum().astype(int).to_dict(),
    }

    return data, report, rejected


class UnderwritingFeatures(TransformerMixin, BaseEstimator):
    """Add deterministic risk features from the approved underwriting inputs."""
    def __init__(self, engineered=False, include_premium=True):
        self.engineered = engineered
        self.include_premium = include_premium

    def fit(self, X, y=None):
        # This transformer does not estimate medians, coefficients, or encodings
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        result = X[FEATURES].copy()

        if self.engineered:
            al = result.coverage_type.eq("AL")
            fleet = result['vehicle_count'].where(result['vehicle_count'].gt(0))
            # A zero/unknown denominator gives NaN rather than an infinite ratio
            result['prior_claims_same_coverage'] = np.where(
                al, result['prior_al_claim_count'], result['prior_apd_claim_count'])
            result['log_prior_loss'] = np.log1p(result['prior_loss_amount'])
            result['log_mileage_per_vehicle'] = np.log1p(result['prior_year_mileage_000'] / fleet)
            result['heavy_vehicle_share'] = result['num_heavy_vehicles'] / fleet
            result['drivers_per_vehicle'] = result['driver_count'] / fleet
            result['al_vehicle_count'] = result['vehicle_count'] * al
            result['al_external_risk'] = result['risk_score_external'] * al
    
        if not self.include_premium:
            result = result.drop(columns='annual_premium')
        
        return result

    def get_feature_names_out(self, input_features=None):
        names = FEATURES + (DERIVED if self.engineered else [])
        return np.asarray([x for x in names if self.include_premium or x != 'annual_premium'])


class PoissonOccurrenceClassifier(ClassifierMixin, BaseEstimator):
    """Fit full-policy claim counts, then convert their Poisson mean to P(N >= 1).

    This challenger assumes a Poisson count distribution. All supplied valid
    periods are 365 days, so no differing policy-time exposure is needed here.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, X, y):
        # Here y contains claim counts, not the notebook's binary evaluation label
        self.regressor_ = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(X, y)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = self.regressor_.n_features_in_
        return self

    def predict_proba(self, X):
        expected_count = self.regressor_.predict(X)
        # P(N = 0) = exp(-lambda). expm1 is stable for small expected counts
        probability = -np.expm1(-expected_count)
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_model(config):
    """Return an unfitted pipeline; all learned preparation stays inside fit()."""
    family = config['family']
    native_categories = config.get('native_categories', False)
    numeric = NUMERIC + (DERIVED if config.get('engineered', False) else [])
    if not config.get('include_premium', True):
        numeric = [col for col in numeric if col != 'annual_premium']
    
    numeric_steps = [
        ('impute', SimpleImputer(strategy='median',
                                 add_indicator=True,
                                 keep_empty_features=True))
    ]
    
    if family in ['logistic', 'poisson']:
        numeric_steps.append(('scale', StandardScaler()))
    if family == 'logistic':
        classifier = LogisticRegression(
            C=config.get("C", 0.2),
            max_iter=3000,
            random_state=SEED
        )
    elif family == 'boosting':
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=config.get('max_iter', 160),
            max_leaf_nodes=config.get('max_leaf_nodes', 8), min_samples_leaf=30,
            l2_regularization=20, early_stopping=False, random_state=SEED,
            categorical_features=(list(range(len(numeric), len(numeric) + len(CATEGORICAL)))
                                  if native_categories else None))
    elif family == 'poisson':
        classifier = PoissonOccurrenceClassifier(alpha=config.get('alpha', 0.1))
    else:
        raise ValueError(f'Unknown model family: {family}')

    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant',
                                 fill_value='Missing',
                                 keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore',
                                 sparse_output=False)),
    ])

    if native_categories:
        if family != 'boosting':
            raise ValueError('Native categorical handling is only available for boosting')
        # Integers identify categories; categorical_features prevents numeric ordering
        # Missing/new labels map to NaN, which histogram boosting handles directly
        categorical = OrdinalEncoder(handle_unknown='use_encoded_value',
                                     unknown_value=np.nan, encoded_missing_value=np.nan)

    preprocessing = ColumnTransformer([
        ('numeric', 'passthrough' if native_categories else Pipeline(numeric_steps), numeric),
        ('categorical', categorical, CATEGORICAL),
    ])

    return Pipeline([
        ('underwriting', UnderwritingFeatures(config.get('engineered', False),
                                              config.get('include_premium', True))),
        ('preprocess', preprocessing), ("classifier", classifier),
    ])


def probability_metrics(y, probabilities):
    """Probability quality and ranking; lower loss/Brier and higher AUC/AP are better."""
    y, p = np.asarray(y), np.asarray(probabilities)

    if p.shape != y.shape or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Probabilities must align with labels and be finite in [0, 1]")

    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if np.any(y == 1) else None,
    }


def training_reference(data):
    """Keep aggregate input statistics for a descriptive scoring quality report."""
    return {
        "categories": {c: sorted(data[c].dropna().unique().tolist()) for c in CATEGORICAL},
        "numeric": {
            c: {
                "mean": float(data[c].mean()) if data[c].notna().any() else None,
                "std": float(data[c].std()) if data[c].notna().sum() > 1 else None,
                "missing_fraction": float(data[c].isna().mean())
            }
            for c in NUMERIC
        },
    }


def score_policies(model, raw, metadata):
    """Preserve input order and return one probability for every input policy."""
    data, quality, _ = prepare_data(raw, training=False)

    durations = (data.policy_expiration_date - data.policy_effective_date).dt.days
    if not durations.isin(metadata["policy_durations_days"]).all():
        raise ValueError("A scoring policy has a term not represented in training")

    # Locate class 1 explicitly instead of relying on an assumed column ordering
    p = model.predict_proba(data[FEATURES])[:, list(model.classes_).index(1)]
    if p.shape != (len(raw),) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid risk scores or incorrect output length")

    reference = metadata["training_reference"]
    quality["unseen_categories"] = {
        c: int((data[c].notna() & ~data[c].isin(reference["categories"][c])).sum())
        for c in CATEGORICAL}
    quality["numeric_mean_shift_in_training_std"] = {
        c: float((data[c].mean() - stats["mean"]) / stats["std"])
        if stats["std"] is not None and stats["std"] > 0 and data[c].notna().any()
        else None for c, stats in reference["numeric"].items()}
    
    output = pd.DataFrame({"policy_id": raw.policy_id.to_numpy(), "risk_score": p})

    return output, quality


def save_model(model, directory, metadata):
    """Persist the entire fitted pipeline with its data and environment contract."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "risk_model.joblib"
    # Save fitted preprocessing together with the classifier, not coefficients alone
    joblib.dump(model, path)
    manifest = dict(metadata, schema_version=2, target="claim_count > 0",
                    score_definition="Probability of at least one claim during the policy period",
                    input_features=FEATURES, random_seed=SEED,
                    versions={"python": platform.python_version(), "pandas": pd.__version__,
                              "numpy": np.__version__, "sklearn": sklearn.__version__,
                              "joblib": joblib.__version__})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def load_model(directory):
    """Load an artifact created by this solution; only load trusted joblib files."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 2 or manifest["input_features"] != FEATURES:
        raise ValueError("Unsupported artifact schema")
    current = {"sklearn": sklearn.__version__, "numpy": np.__version__,
               "pandas": pd.__version__, "joblib": joblib.__version__}
    if any(manifest["versions"][key] != value for key, value in current.items()):
        raise ValueError("Use the dependency versions recorded in manifest.json")
    if manifest["versions"]["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
        raise ValueError("Use the Python major/minor version recorded in manifest.json")
    return joblib.load(directory / "risk_model.joblib"), manifest
