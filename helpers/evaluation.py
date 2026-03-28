import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)

def find_optimal_threshold(y_true, y_proba):
    """
    Find threshold where F1 is maximized
    We sort using F1 scores:
    Formula:
        $ 2 * \frac{precision * recall} {precision + recall}

    Returns:
        optiomal_threshold, best_f1_score

    """

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)
    _f1_scores = 2.0 * (precision_arr * recall_arr) \
                    / (precision_arr + recall_arr + 1e-8) # offset
    optimal_index = np.argmax(_f1_scores[:-1])

    return thresholds[optimal_index], _f1_scores[optimal_index]

def evaluate_model(name, model, x_test, y_test, use_proba = True):
    """
    Evaluation of model

    Retunrs:
        Dict with all metrics
    """

    # Get probabilities
    if use_proba:
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_proba = model.predict(x_test)

    # Find threshold
    optimal_threshold , _ = find_optimal_threshold(y_test, y_proba)

    # Predictions at said threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Get metrics
    results = {
        'Model'         : name,
        'Opt_Threshold' : optimal_threshold,
        'Precision'     : precision_score(y_test, y_pred, zero_division=0),
        'Recall'        : recall_score(y_test, y_pred, zero_division=0),
        'F1_Score'      : f1_score(y_test, y_pred, zero_division=0),
        'AUC_ROC'       : roc_auc_score(y_test, y_proba),
        'AUC_PR'        : average_precision_score(y_test, y_proba)
    }
    
    return results, y_proba, y_pred

def display_results(metrics):
    """
    Display comparison table sorted by F1 Score, the harmony between precision and recall rate.

    It ensures the model has both high precision (few false positives) and high recall (few false negatives).

    """
    df = pd.DataFrame(metrics)
    df = df.sort_values('F1_Score', ascending=False)
    
    print("\n" + "="*90)
    print("MODEL COMPARISON (sorted by F1 Score)")
    print("="*90)
    
    # Format for display
    display_df = df.copy()
    for col in ['Opt_Threshold', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC', 'AUC_PR']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    
    best = df.iloc[0]
    print(f"\Optimal Model: {best['Model']} (F1: {best['F1_Score']:.4f})")
    
    return df