import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ml_pipeline.config import ML_FEATURES, TARGET_COL, FP_COST, FN_COST
from src.ml_pipeline.feature_engineer import prepare_ml_data

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed, skipping")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("lightgbm not installed, skipping")


def load_data():
    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    X_train, y_train = prepare_ml_data(train_df)
    X_test, y_test = prepare_ml_data(test_df)
    return X_train, y_train, X_test, y_test


def get_models():
    models = {}

    # baseline — logistic regression
    models['LogReg'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    # random forest — our main model
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=-1
    )

    # random forest + smote — compare vs just class_weight
    models['RF_SMOTE'] = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ))
    ])

    # gradient boosting
    models['GradBoost'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42
    )

    if HAS_XGB:
        # xgboost — usually beats RF on tabular data
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=9,  # roughly (1-return_rate)/return_rate
            eval_metric='aucpr',
            random_state=42, n_jobs=-1, verbosity=0
        )

    if HAS_LGB:
        # lightgbm — fastest of the bunch
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        )

    return models


def evaluate_model_at_threshold(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        'pr_auc': average_precision_score(y_test, y_proba),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'business_cost': (fp * FP_COST) + (fn * FN_COST),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def cross_validate_models(models, X_train, y_train, cv=5):
    print(f"\n{'='*60}")
    print(f"  Stratified {cv}-Fold Cross Validation (PR-AUC)")
    print(f"{'='*60}")

    cv_results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n  {name}...")
        try:
            scores = cross_val_score(
                model, X_train, y_train,
                scoring='average_precision',
                cv=skf, n_jobs=1  # n_jobs=1 to avoid nested parallelism issues
            )
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            print(f"    PR-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            cv_results[name] = {'mean': 0, 'std': 0, 'scores': []}

    return cv_results


def find_fpc_threshold(model, X_test, y_test):
    """Find optimal threshold by minimizing FP cost + FN cost."""
    y_proba = model.predict_proba(X_test)[:, 1]
    best_thresh = 0.5
    min_cost = float('inf')

    for t in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cost = (fp * FP_COST) + (fn * FN_COST)
        if cost < min_cost:
            min_cost = cost
            best_thresh = round(t, 2)

    return best_thresh, min_cost


def plot_model_comparison(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Comparison — AI Return-Risk Scorer', fontsize=16, fontweight='bold', y=1.02)

    colors = ['#4361ee', '#f72585', '#7209b7', '#3a0ca3', '#4cc9f0', '#4895ef']
    models_list = results_df['model'].tolist()

    # PR-AUC comparison
    ax = axes[0]
    bars = ax.barh(models_list, results_df['pr_auc'], color=colors[:len(models_list)], alpha=0.85)
    ax.set_xlabel('PR-AUC (higher is better)', fontsize=11)
    ax.set_title('Precision-Recall AUC', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, results_df['pr_auc']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    ax.axvline(x=results_df['pr_auc'].max(), color='red', linestyle='--', alpha=0.4, linewidth=1)

    # F1 Score comparison
    ax = axes[1]
    bars = ax.barh(models_list, results_df['f1'], color=colors[:len(models_list)], alpha=0.85)
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score at Optimal Threshold', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, results_df['f1']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)

    # Business cost comparison
    ax = axes[2]
    bars = ax.barh(models_list, results_df['business_cost'] / 1000, color=colors[:len(models_list)], alpha=0.85)
    ax.set_xlabel('Business Cost (Rs. thousands, lower is better)', fontsize=11)
    ax.set_title('Total Business Cost at Optimal Threshold', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, results_df['business_cost'] / 1000):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'Rs.{val:.1f}K', va='center', fontsize=9)
    ax.axvline(x=results_df['business_cost'].min() / 1000, color='green', linestyle='--', alpha=0.4, linewidth=1)

    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  Model Comparison — Return-Risk Scorer")
    print("=" * 60)
    print(f"\n  FP cost: Rs.{FP_COST}, FN cost: Rs.{FN_COST}")

    X_train, y_train, X_test, y_test = load_data()
    print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Return rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    models = get_models()
    print(f"\n  Models to compare: {list(models.keys())}")

    # Cross-validation first
    cv_results = cross_validate_models(models, X_train, y_train, cv=5)

    # Train each model on full training set and evaluate on test
    print(f"\n{'='*60}")
    print("  Test Set Evaluation at FPC-Optimal Threshold")
    print(f"{'='*60}")

    all_results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model

            # find optimal threshold per model
            best_thresh, min_cost = find_fpc_threshold(model, X_test, y_test)
            metrics = evaluate_model_at_threshold(model, X_test, y_test, best_thresh)

            row = {
                'model': name,
                'cv_pr_auc_mean': cv_results.get(name, {}).get('mean', 0),
                'cv_pr_auc_std': cv_results.get(name, {}).get('std', 0),
                'optimal_threshold': best_thresh,
                **metrics
            }
            all_results.append(row)

            print(f"    Optimal threshold: {best_thresh}")
            print(f"    PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"    F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"    Business Cost: Rs.{metrics['business_cost']:,.0f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('pr_auc', ascending=False).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("  Final Rankings (by PR-AUC)")
    print(f"{'='*60}")
    print(results_df[['model', 'cv_pr_auc_mean', 'pr_auc', 'f1', 'optimal_threshold', 'business_cost']].to_string(index=False))

    best_model_name = results_df.iloc[0]['model']
    print(f"\n  Best model: {best_model_name}")
    print(f"  CV PR-AUC: {results_df.iloc[0]['cv_pr_auc_mean']:.4f} ± {results_df.iloc[0]['cv_pr_auc_std']:.4f}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    results_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved: {results_path}")

    plot_model_comparison(results_df, os.path.join(output_dir, 'plots'))

    # Save best model if it beats our current RF
    current_model_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'models', 'random_forest.pkl')
    if os.path.exists(current_model_path):
        with open(current_model_path, 'rb') as f:
            current = pickle.load(f)
        current_pr_auc = current.get('pr_auc', 0)
        best_pr_auc = results_df.iloc[0]['pr_auc']

        print(f"\n  Current deployed model PR-AUC: {current_pr_auc:.4f}")
        print(f"  Best from comparison PR-AUC:   {best_pr_auc:.4f}")

        if best_pr_auc > current_pr_auc + 0.005:
            print(f"  New best model ({best_model_name}) beats current by {best_pr_auc - current_pr_auc:.4f}")
            print("  Run train_model.py with the new estimator to deploy it.")
        else:
            print("  Current deployed model is still the best or equivalent.")

    print(f"\n{'='*60}")
    print("  Model comparison complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
