import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ml_pipeline.config import (
    ML_FEATURES, TARGET_COL,
    N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT,
    MIN_SAMPLES_LEAF, CLASS_WEIGHT, RANDOM_STATE, N_JOBS,
    FP_COST, FN_COST,
    THRESH_MIN, THRESH_MAX, THRESH_STEP, DEFAULT_THRESH,
)
from src.ml_pipeline.feature_engineer import prepare_ml_data


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    model.fit(X_train, y_train)
    return model


def find_best_threshold(model, X_test, y_test):
    # sweep thresholds and pick the one with lowest business cost
    # formula: total_cost = FP * fp_cost + FN * fn_cost
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(THRESH_MIN, THRESH_MAX + THRESH_STEP, THRESH_STEP)

    results = []
    min_cost = float('inf')
    best_thresh = DEFAULT_THRESH

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        cost = (fp * FP_COST) + (fn * FN_COST)
        results.append({
            'threshold': round(t, 2),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'fp_cost': fp * FP_COST,
            'fn_cost': fn * FN_COST,
            'total_cost': cost,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        })

        if cost < min_cost:
            min_cost = cost
            best_thresh = round(t, 2)

    return best_thresh, min_cost, pd.DataFrame(results)


def main():
    print("Training model...")

    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')

    if not os.path.exists(train_path):
        print("train.csv not found, run feature_engineer.py first")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"Return rate in train: {train_df[TARGET_COL].mean():.2%}")

    X_train, y_train = prepare_ml_data(train_df)
    X_test, y_test = prepare_ml_data(test_df)

    print("Fitting random forest...")
    model = train_model(X_train, y_train)
    print("Done training")

    # feature importance - just curious which features matter most
    importances = pd.Series(model.feature_importances_, index=ML_FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    for feat, imp in importances.items():
        print(f"  {feat}: {imp:.4f}")

    print("\nFinding optimal threshold using FPC...")
    print(f"FP cost: {FP_COST}, FN cost: {FN_COST}")
    best_thresh, min_cost, costs_df = find_best_threshold(model, X_test, y_test)

    print(f"Best threshold: {best_thresh}")
    print(f"Min business cost: {min_cost:,.0f}")

    # compare with default 0.5
    default_row = costs_df[costs_df['threshold'] == DEFAULT_THRESH]
    if not default_row.empty:
        default_cost = default_row.iloc[0]['total_cost']
        print(f"Cost at 0.5 threshold: {default_cost:,.0f}")
        print(f"Savings: {default_cost - min_cost:,.0f}")

    # final metrics at best threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_thresh).astype(int)

    print(f"\nClassification report at threshold={best_thresh}:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Return']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"PR-AUC: {pr_auc:.4f}")

    # save model
    print("\nCreating SHAP explainer (takes a bit)...")
    import shap
    background = X_train.sample(min(500, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model, data=background)
    print("SHAP explainer done")

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'api', 'models')
    os.makedirs(model_dir, exist_ok=True)

    # save everything we need for inference
    artifact = {
        'model': model,
        'optimal_threshold': best_thresh,
        'feature_names': ML_FEATURES,
        'pr_auc': pr_auc,
        'min_business_cost': min_cost,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
    }

    model_save_path = os.path.join(model_dir, 'random_forest.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"Model saved: {model_save_path}")

    explainer_save_path = os.path.join(model_dir, 'shap_explainer.pkl')
    with open(explainer_save_path, 'wb') as f:
        pickle.dump(explainer, f)
    print(f"SHAP explainer saved: {explainer_save_path}")

    # save threshold costs csv for analysis
    costs_path = os.path.join(processed_dir, 'threshold_costs.csv')
    costs_df.to_csv(costs_path, index=False)
    print(f"Threshold costs saved: {costs_path}")

    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
