# config.py - all the constants and settings for the project
# keeping everything in one place so I don't have to hunt for numbers

# business cost stuff - asked my mentor what values to use
# blocking a real customer hurts more than missing a fraud (CLV reason)
FP_COST = 350   # cost when we block a genuine order
FN_COST = 1200  # cost when a fraud slips through (shipping + reverse + inventory)

# dataset params
NUM_TRANSACTIONS = 50000
RETURN_RATE = 0.10      # roughly 10% returns in indian ecom
FRAUD_RETURN_RATE = 0.05

# date range for the synthetic data
DATA_START = "2025-01-01"
DATA_END = "2025-12-31"

# train on jan-oct, test on nov-dec
# doing chronological split instead of random to avoid data leakage
TRAIN_END = "2025-10-31"
TEST_START = "2025-11-01"

# random forest params - tuned these by trial and error
N_ESTIMATORS = 200
MAX_DEPTH = 12
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
CLASS_WEIGHT = "balanced"  # needed because only 10% are returns
RANDOM_STATE = 42
N_JOBS = -1

# threshold sweep range
THRESH_MIN = 0.10
THRESH_MAX = 0.90
THRESH_STEP = 0.01
DEFAULT_THRESH = 0.50

# score to action mapping
# these cutoffs i decided after looking at the cost curve output
ACTION_MAP = {
    "ALLOW": 0.30,
    "VERIFY_OTP": 0.60,
    "MANDATE_PREPAID": 0.80,
    "BLOCK": 1.00,
}

# the 11 features the model actually uses
# order matters - model was trained with this exact order
ML_FEATURES = [
    "geo_distance_km",
    "is_disposable_email",
    "is_cod",
    "transaction_velocity_24h",
    "transaction_velocity_7d",
    "avg_order_value_ratio",
    "account_age_days",
    "is_high_risk_category",
    "hour_of_day",
    "is_first_order",
    "return_history_rate",
]

TARGET_COL = "is_return"

# electronics and fashion have way higher return rates from the data analysis
RISKY_CATEGORIES = ["electronics", "fashion", "mobile_accessories", "footwear"]

# weights for the rule engine (for cold start users with no history)
RULE_WEIGHTS = {
    "disposable_email": 0.30,
    "geo_mismatch": 0.25,
    "high_value_cod": 0.20,
    "risky_category": 0.10,
    "late_night": 0.10,
    "new_account": 0.15,
}

GEO_THRESHOLD_KM = 500
COD_VALUE_THRESHOLD = 5000
NEW_ACCOUNT_DAYS = 7
LATE_HOURS = range(1, 6)  # 1am to 5am

SHAP_TOP_N = 5

# model file paths
MODEL_PATH = "src/api/models/random_forest.pkl"
SHAP_PATH = "src/api/models/shap_explainer.pkl"

PAYMENT_METHODS = ["COD", "UPI", "CARD", "WALLET"]

CATEGORIES = [
    "electronics", "fashion", "mobile_accessories", "footwear",
    "home_kitchen", "beauty", "books", "grocery", "sports", "toys",
]
