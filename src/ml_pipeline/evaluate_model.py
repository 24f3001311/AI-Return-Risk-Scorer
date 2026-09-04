"""
Model Evaluation Module
=======================
Generates comprehensive evaluation metrics and visualizations
for the trained Random Forest model.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ml_pipeline.config import (
    ML_FEATURES, TARGET_COL,
    FP_COST, FN_COST,
)
from src.ml_pipeline.feature_engineer import prepare_ml_data


def load_model():
    """Load the trained model artifact."""
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'api', 'models', 'random_forest.pkl'
    )
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    return artifact


def generate_evaluation_plots(y_test, y_proba, optimal_threshold, output_dir):
    """
    Generate and save evaluation plots.
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities
        optimal_threshold: FPC-optimal threshold
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(10, 7))
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    ax.plot(recall, precision, linewidth=2.5, color='#e94560',
            label=f'PR Curve (AUC = {pr_auc:.4f})')
    
    # Mark optimal threshold
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score
    opt_precision = precision_score(y_test, y_pred_opt)
    opt_recall = recall_score(y_test, y_pred_opt)
    ax.scatter([opt_recall], [opt_precision], s=200, c='#0f3460', 
               zorder=5, label=f'Optimal Threshold ({optimal_threshold})',
               edgecolors='white', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve with FPC-Optimal Threshold', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=150)
    plt.close()
    
    # 2. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pred = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Safe', 'Return'],
        yticklabels=['Safe', 'Return'],
        annot_kws={'size': 16},
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title(f'Confusion Matrix (threshold={optimal_threshold})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # 3. FPC Cost vs Threshold
    costs_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'processed', 'threshold_costs.csv'
    )
    if os.path.exists(costs_path):
        costs_df = pd.read_csv(costs_path)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(costs_df['threshold'], costs_df['total_cost'] / 1000, 
                linewidth=2.5, color='#e94560', label='Total Business Cost')
        ax.plot(costs_df['threshold'], costs_df['fp_cost'] / 1000, 
                linewidth=1.5, linestyle='--', color='#ffa500', label='FP Cost (Orders Wrongly Blocked)')
        ax.plot(costs_df['threshold'], costs_df['fn_cost'] / 1000, 
                linewidth=1.5, linestyle='--', color='#53d769', label='FN Cost (RTOs Missed)')
        
        ax.axvline(x=optimal_threshold, color='#0f3460', linestyle=':', linewidth=2,
                   label=f'Optimal Threshold ({optimal_threshold})')
        
        ax.set_xlabel('Decision Threshold', fontsize=14)
        ax.set_ylabel('Business Cost ( thousands)', fontsize=14)
        ax.set_title('False-Positive Cost Matrix  Threshold Optimization', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fpc_threshold_curve.png'), dpi=150)
        plt.close()
    
    # 4. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    artifact = load_model()
    model = artifact['model']
    importances = pd.Series(
        model.feature_importances_, index=ML_FEATURES
    ).sort_values(ascending=True)
    
    colors = ['#e94560' if v > importances.median() else '#0f3460' for v in importances.values]
    importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Feature Importance (Gini)', fontsize=14)
    ax.set_title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    
    # 5. Score Distribution
    fig, ax = plt.subplots(figsize=(10, 7))
    safe_scores = y_proba[y_test == 0]
    return_scores = y_proba[y_test == 1]
    
    ax.hist(safe_scores, bins=50, alpha=0.6, color='#53d769', label='Safe Orders', density=True)
    ax.hist(return_scores, bins=50, alpha=0.6, color='#e94560', label='Returns/RTO', density=True)
    ax.axvline(x=optimal_threshold, color='#0f3460', linestyle='--', linewidth=2,
               label=f'Threshold ({optimal_threshold})')
    
    ax.set_xlabel('Risk Score', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Risk Score Distribution by Class', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=150)
    plt.close()
    
    print(f"    Plots saved to: {output_dir}")


def plot_correlation_heatmap(train_df, output_dir):
    """Correlation heatmap of all ML features + target."""
    os.makedirs(output_dir, exist_ok=True)

    cols = ML_FEATURES + [TARGET_COL]
    corr_matrix = train_df[cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # upper triangle mask

    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 8},
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title('Feature Correlation Matrix\n(lower triangle only, diagonal = 1.0)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Correlation heatmap saved: {path}")

    # Print highly correlated feature pairs (>0.7) - potential redundancy
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            val = abs(corr_matrix.iloc[i, j])
            if val > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], round(val, 4)))

    if high_corr:
        print("    High correlations (>0.7) — potential redundant features:")
        for f1, f2, v in high_corr:
            print(f"      {f1} <-> {f2}: {v}")
    else:
        print("    No highly correlated feature pairs found (>0.7) — good!")


def feature_selection_analysis(X_train, y_train, model, X_test, y_test, output_dir):
    """Mutual information + permutation importance — identifies which features actually matter."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Mutual information (filter method — model-agnostic)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=ML_FEATURES).sort_values(ascending=False)

    print("\n    Mutual Information Scores (feature relevance, model-agnostic):")
    for feat, score in mi_series.items():
        bar = '#' * int(score * 50)
        print(f"      {feat:35s} {score:.4f}  {bar}")

    # 2. Permutation importance (wrapper method — model-specific, more reliable)
    print("\n    Computing permutation importance (takes ~30s)...")
    perm_imp = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42,
        scoring='average_precision'
    )
    perm_series = pd.Series(
        perm_imp.importances_mean, index=ML_FEATURES
    ).sort_values(ascending=False)

    print("    Permutation Importance (drop in PR-AUC when feature shuffled):")
    for feat, score in perm_series.items():
        bar = '#' * max(0, int(score * 200))
        marker = ' <- LOW IMPORTANCE' if score < 0.001 else ''
        print(f"      {feat:35s} {score:+.4f}  {bar}{marker}")

    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Feature Selection Analysis', fontsize=15, fontweight='bold')

    # MI plot
    colors_mi = ['#e94560' if v > mi_series.median() else '#4361ee' for v in mi_series.values]
    mi_series.sort_values().plot(kind='barh', ax=ax1, color=colors_mi[::-1], alpha=0.85)
    ax1.set_xlabel('Mutual Information Score', fontsize=11)
    ax1.set_title('Mutual Information\n(filter method, model-agnostic)', fontsize=12, fontweight='bold')
    ax1.axvline(x=mi_series.median(), color='gray', linestyle='--', alpha=0.6, label='Median')
    ax1.legend()

    # Permutation importance plot
    perm_sorted = perm_series.sort_values()
    colors_perm = ['#e94560' if v > 0.001 else '#aaa' for v in perm_sorted.values]
    perm_sorted.plot(kind='barh', ax=ax2, color=colors_perm, alpha=0.85)
    ax2.set_xlabel('Mean Drop in PR-AUC (permutation)', fontsize=11)
    ax2.set_title('Permutation Importance\n(wrapper method, model-specific)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'feature_selection_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    Feature selection plot saved: {path}")

    # Warn about potentially useless features
    useless = perm_series[perm_series < 0].index.tolist()
    if useless:
        print(f"\n    WARNING: These features HURT performance when permuted (negative importance):")
        for f in useless:
            print(f"      - {f}")
        print("    Consider removing these in next iteration.")

    return mi_series, perm_series


def main():
    print("=" * 60)
    print("  AI Return-Risk Scorer — Model Evaluation")
    print("=" * 60)

    artifact = load_model()
    model = artifact['model']
    optimal_threshold = artifact['optimal_threshold']

    print(f"\n Model loaded:")
    print(f"   Optimal Threshold: {optimal_threshold}")
    print(f"   PR-AUC: {artifact['pr_auc']:.4f}")
    print(f"   Training Samples: {artifact['training_samples']:,}")

    # Load test and train data
    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))

    X_test, y_test = prepare_ml_data(test_df)
    X_train, y_train = prepare_ml_data(train_df)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= optimal_threshold).astype(int)

    print(f"\n Classification Report (threshold={optimal_threshold}):")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Return']))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fp_cost = fp * FP_COST
    fn_cost = fn * FN_COST
    total_cost = fp_cost + fn_cost

    print(f" Business Impact Analysis:")
    print(f"   True Positives (Returns Caught): {tp:,}")
    print(f"   False Positives (Good Orders Blocked): {fp:,}")
    print(f"   False Negatives (Returns Missed): {fn:,}")
    print(f"   True Negatives (Safe Orders Allowed): {tn:,}")
    print(f"")
    print(f"   FP Cost (Rs.{FP_COST}/order): Rs.{fp_cost:,.0f}")
    print(f"   FN Cost (Rs.{FN_COST}/order): Rs.{fn_cost:,.0f}")
    print(f"   Total Business Cost: Rs.{total_cost:,.0f}")

    output_dir = os.path.join(processed_dir, 'plots')

    print(f"\n Generating evaluation plots...")
    generate_evaluation_plots(y_test.values, y_proba, optimal_threshold, output_dir)

    # NEW: correlation heatmap
    print(f"\n Generating correlation heatmap...")
    plot_correlation_heatmap(train_df, output_dir)

    # NEW: feature selection analysis (MI + permutation importance)
    print(f"\n Running feature selection analysis...")
    mi_scores, perm_scores = feature_selection_analysis(X_train, y_train, model, X_test, y_test, output_dir)

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n ROC-AUC Score: {roc_auc:.4f}")
    print(f" PR-AUC Score: {average_precision_score(y_test, y_proba):.4f}")
    print(f" F1 Score: {f1_score(y_test, y_pred):.4f}")

    print(f"\n{'=' * 60}")
    print("   Model evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

