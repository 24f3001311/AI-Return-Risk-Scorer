import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.ml_pipeline.config import (
    ML_FEATURES, RISKY_CATEGORIES, SHAP_TOP_N, ACTION_MAP
)
from src.api.schemas.transaction import TransactionPayload
from src.api.schemas.response import RiskReason
from src.api.services.geo_utils import calculate_geo_distance


_model_artifact = None
_shap_explainer = None
_disposable_domains = None


def _get_disposable_domains():
    global _disposable_domains
    if _disposable_domains is None:
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'data', 'reference', 'disposable_email_domains.txt'
        )
        if os.path.exists(path):
            with open(path, 'r') as f:
                _disposable_domains = set(line.strip().lower() for line in f if line.strip())
        else:
            _disposable_domains = set()
    return _disposable_domains


def load_model():
    global _model_artifact, _shap_explainer

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest.pkl')
    explainer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'shap_explainer.pkl')

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            _model_artifact = pickle.load(f)
        print(f"Model loaded. threshold={_model_artifact['optimal_threshold']}, pr_auc={_model_artifact['pr_auc']:.4f}")
    else:
        print(f"Model file not found: {model_path}")
        _model_artifact = None

    if os.path.exists(explainer_path):
        with open(explainer_path, 'rb') as f:
            _shap_explainer = pickle.load(f)
        print("SHAP explainer loaded")
    else:
        print("SHAP explainer not found, explanations will be limited")
        _shap_explainer = None


def is_model_loaded():
    return _model_artifact is not None


def get_optimal_threshold():
    if _model_artifact:
        return _model_artifact['optimal_threshold']
    return 0.5


def build_feature_vector(payload: TransactionPayload) -> pd.DataFrame:
    domains = _get_disposable_domains()
    email_domain = payload.email.split('@')[-1].lower()

    geo_dist = calculate_geo_distance(payload.billing_pincode, payload.shipping_pincode)

    return_rate = 0.0
    if payload.total_past_orders > 0:
        return_rate = payload.past_returns / payload.total_past_orders

    # avg_order_value_ratio dynamically compares current order with past user history
    # For MVP, defaulting to 1.0 (average) if historical API is unavailable
    avg_ratio = 1.0

    features = {
        'geo_distance_km': geo_dist,
        'is_disposable_email': int(email_domain in domains),
        'is_cod': int(payload.payment_method == "COD"),
        'transaction_velocity_24h': payload.transaction_velocity_24h,
        'transaction_velocity_7d': payload.transaction_velocity_7d,
        'avg_order_value_ratio': avg_ratio,
        'account_age_days': payload.account_age_days,
        'is_high_risk_category': int(payload.product_category.lower() in [c.lower() for c in RISKY_CATEGORIES]),
        'hour_of_day': pd.Timestamp.now().hour,
        'is_first_order': int(payload.total_past_orders == 0),
        'return_history_rate': return_rate,
    }

    return pd.DataFrame([features])[ML_FEATURES]


def predict_with_shap(payload: TransactionPayload):
    if _model_artifact is None:
        raise RuntimeError("Model not loaded")

    model = _model_artifact['model']
    X = build_feature_vector(payload)

    risk_score = float(model.predict_proba(X)[:, 1][0])
    reasons = []

    if _shap_explainer is not None:
        try:
            shap_vals = _shap_explainer.shap_values(X)

            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                sv = shap_vals[0, :, 1]
            else:
                sv = shap_vals[0]

            pairs = list(zip(ML_FEATURES, sv, X.iloc[0].values))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            for feat, shap_val, feat_val in pairs[:SHAP_TOP_N]:
                direction = "increases" if shap_val > 0 else "decreases"
                desc = _shap_desc(feat, feat_val, shap_val)
                reasons.append(RiskReason(
                    feature=feat,
                    value=float(feat_val),
                    contribution=round(float(shap_val), 4),
                    description=desc,
                ))

        except Exception as e:
            reasons.append(RiskReason(
                feature="model_inference",
                value=risk_score,
                contribution=risk_score,
                description=f"ML model scored this transaction at {risk_score:.2f}. SHAP unavailable: {str(e)}",
            ))
    else:
        reasons.append(RiskReason(
            feature="model_inference",
            value=risk_score,
            contribution=risk_score,
            description=f"ML model scored this at {risk_score:.2f}",
        ))

    return risk_score, reasons


def _shap_desc(feature, value, shap_value):
    direction = "increases" if shap_value > 0 else "decreases"

    desc_map = {
        'geo_distance_km': f"Geo distance {value:.0f}km {direction} risk",
        'is_disposable_email': f"{'Disposable' if value == 1 else 'Legit'} email {direction} risk",
        'is_cod': f"{'COD' if value == 1 else 'Prepaid'} payment {direction} risk",
        'transaction_velocity_24h': f"{value:.0f} orders in 24h {direction} risk",
        'transaction_velocity_7d': f"{value:.0f} orders in 7d {direction} risk",
        'avg_order_value_ratio': f"Order value is {value:.2f}x of usual, {direction} risk",
        'account_age_days': f"Account is {value:.0f} days old, {direction} risk",
        'is_high_risk_category': f"{'High-risk' if value == 1 else 'Normal'} category {direction} risk",
        'hour_of_day': f"Ordered at {int(value)}:00, {direction} risk",
        'is_first_order': f"{'First order' if value == 1 else 'Repeat customer'} {direction} risk",
        'return_history_rate': f"Return rate {value:.1%} {direction} risk",
    }

    return desc_map.get(feature, f"{feature}={value:.4f} {direction} risk")


def score_to_action(score: float) -> str:
    if score < ACTION_MAP['ALLOW']:
        return 'ALLOW'
    elif score < ACTION_MAP['VERIFY_OTP']:
        return 'VERIFY_OTP'
    elif score < ACTION_MAP['MANDATE_PREPAID']:
        return 'MANDATE_PREPAID'
    else:
        return 'BLOCK'
