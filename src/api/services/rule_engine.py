import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.ml_pipeline.config import (
    RULE_WEIGHTS, GEO_THRESHOLD_KM, COD_VALUE_THRESHOLD,
    NEW_ACCOUNT_DAYS, LATE_HOURS, RISKY_CATEGORIES
)
from src.api.schemas.transaction import TransactionPayload
from src.api.schemas.response import RiskReason
from src.api.services.geo_utils import calculate_geo_distance


_disposable_domains = None


def _get_domains():
    global _disposable_domains
    if _disposable_domains is None:
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'data', 'reference', 'disposable_email_domains.txt'
        )
        if os.path.exists(path):
            with open(path, 'r') as f:
                _disposable_domains = set(l.strip().lower() for l in f if l.strip())
        else:
            _disposable_domains = set()
    return _disposable_domains


def evaluate_with_rules(payload: TransactionPayload):
    score = 0.0
    reasons = []

    # rule 1: disposable email
    domain = payload.email.split('@')[-1].lower()
    if domain in _get_domains():
        w = RULE_WEIGHTS['disposable_email']
        score += w
        reasons.append(RiskReason(
            feature="is_disposable_email",
            value=1.0,
            contribution=w,
            description=f"Email domain '{domain}' is a known disposable provider",
        ))

    # rule 2: billing vs shipping distance
    dist = calculate_geo_distance(payload.billing_pincode, payload.shipping_pincode)
    if dist > GEO_THRESHOLD_KM:
        w = RULE_WEIGHTS['geo_mismatch']
        score += w
        reasons.append(RiskReason(
            feature="geo_distance_km",
            value=dist,
            contribution=w,
            description=f"Billing and shipping are {dist:.0f}km apart",
        ))

    # rule 3: high value cod - risky because customer can just reject
    if payload.order_value > COD_VALUE_THRESHOLD and payload.payment_method == "COD":
        w = RULE_WEIGHTS['high_value_cod']
        score += w
        reasons.append(RiskReason(
            feature="is_cod",
            value=payload.order_value,
            contribution=w,
            description=f"COD order worth {payload.order_value:,.0f} (threshold {COD_VALUE_THRESHOLD:,})",
        ))

    # rule 4: risky category
    if payload.product_category.lower() in [c.lower() for c in RISKY_CATEGORIES]:
        w = RULE_WEIGHTS['risky_category']
        score += w
        reasons.append(RiskReason(
            feature="is_high_risk_category",
            value=1.0,
            contribution=w,
            description=f"Category '{payload.product_category}' has high return rates",
        ))

    # rule 5: late night order
    hour = datetime.now().hour
    if hour in LATE_HOURS:
        w = RULE_WEIGHTS['late_night']
        score += w
        reasons.append(RiskReason(
            feature="hour_of_day",
            value=float(hour),
            contribution=w,
            description=f"Order at {hour}:00 (late night 1am-5am)",
        ))

    # rule 6: brand new account
    if payload.account_age_days < NEW_ACCOUNT_DAYS:
        w = RULE_WEIGHTS['new_account']
        score += w
        reasons.append(RiskReason(
            feature="account_age_days",
            value=float(payload.account_age_days),
            contribution=w,
            description=f"Account only {payload.account_age_days} days old",
        ))

    score = min(score, 1.0)

    if not reasons:
        reasons.append(RiskReason(
            feature="baseline",
            value=0.0,
            contribution=0.0,
            description="No risk signals found",
        ))

    return score, reasons
