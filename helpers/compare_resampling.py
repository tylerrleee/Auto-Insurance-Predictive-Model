"""
Compare Resampling Techniques for Logistic Regression

This module provides functions to evaluate different resampling strategies
for handling class imbalance in binary classification.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
 
 
def compare_resampling_techniques(X_train, y_train, X_test, y_test, random_state=42):
    """
    Compare multiple resampling techniques using Logistic Regression.
    
    Parameters
        X_train : array-like
            Training features (should be scaled)
        y_train : array-like
            Training labels
        X_test : array-like
            Test features (should be scaled)
        y_test : array-like
            Test labels
        random_state : int
            Random seed for reproducibility
        
    Returns
        results_df : pd.DataFrame
            Comparison table with metrics for each technique
        detailed_results : dict
            Dictionary containing predictions and probabilities for each technique
    """
    
    # Define resampling techniques to compare
    techniques = {
        'Baseline (No Resampling)': None,
        'SMOTE': SMOTE(random_state=random_state),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=random_state),
        'ADASYN': ADASYN(random_state=random_state),
        'SMOTE + Tomek': SMOTETomek(random_state=random_state)
    }
    
    results = []
    detailed_results = {}
    
    for name, sampler in techniques.items():
        print(f"Training: {name}...")
        
        # Apply resampling (or use original data for baseline)
        if sampler is not None:
            try:
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            except Exception as e:
                print(f" {name} failed: {e}")
                continue
        else:
            X_resampled, y_resampled = X_train, y_train
        
        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
        model.fit(X_resampled, y_resampled)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics (for the minority/positive class)
        metrics = {
            'Technique': name,
            'Train Samples': len(y_resampled),
            'Minority Samples': sum(y_resampled == 1),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'AUC-PR': average_precision_score(y_test, y_proba)
        }
        
        results.append(metrics)
        
        # Store detailed results for further analysis
        detailed_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'X_resampled_shape': X_resampled.shape,
            'class_distribution': pd.Series(y_resampled).value_counts().to_dict()
        }

        # Gives report of each 
        #print(classification_report(y_test, y_pred, target_names=['No Claim', 'Claim']))
        print(f" Done - F1: {metrics['F1 Score']:.4f}, AUC-PR: {metrics['AUC-PR']:.4f}")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by F1 Score (or your preferred metric)
    results_df = results_df.sort_values('F1 Score', ascending=False)
    
    return results_df, detailed_results
 
 
def display_comparison(results_df, highlight_best=True):
    """
    Display the comparison table with optional highlighting.
    
    Parameters
        results_df : pd.DataFrame
            Output from compare_resampling_techniques
        highlight_best : bool
            Whether to print the best technique
    """
    print("\n" + "=" * 90)
    print("RESAMPLING TECHNIQUES COMPARISON")
    print("=" * 90)
    
    # Format for display
    display_df = results_df.copy()
    display_df['Train Samples'] = display_df['Train Samples'].apply(lambda x: f"{x:,}")
    display_df['Minority Samples'] = display_df['Minority Samples'].apply(lambda x: f"{x:,}")
    
    for col in ['Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'AUC-PR']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    
    if highlight_best:
        best = results_df.iloc[0]
        print(f"\nBest by F1 Score: {best['Technique']} (F1: {best['F1 Score']:.4f})")
        
        # Also show best by AUC-PR (often more relevant for imbalanced data)
        best_auc_pr = results_df.loc[results_df['AUC-PR'].idxmax()]
        if best_auc_pr['Technique'] != best['Technique']:
            print(f"Best by AUC-PR: {best_auc_pr['Technique']} (AUC-PR: {best_auc_pr['AUC-PR']:.4f})")
 
 
def plot_comparison(results_df, detailed_results, y_test):
    """
    Create visualization comparing all techniques.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from compare_resampling_techniques
    detailed_results : dict
        Detailed results dictionary
    y_test : array-like
        Test labels
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    techniques = results_df['Technique'].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(techniques)))
    
    # 1. Bar chart: Precision, Recall, F1
    ax1 = axes[0, 0]
    x = np.arange(len(techniques))
    width = 0.25
    
    ax1.bar(x - width, results_df['Precision'], width, label='Precision', color='#3498db')
    ax1.bar(x, results_df['Recall'], width, label='Recall', color='#e74c3c')
    ax1.bar(x + width, results_df['F1 Score'], width, label='F1 Score', color='#2ecc71')
    ax1.set_xticks(x)
    ax1.set_xticklabels(techniques, rotation=45, ha='right')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision / Recall / F1 Comparison')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Bar chart: AUC-ROC vs AUC-PR
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, results_df['AUC-ROC'], width, label='AUC-ROC', color='#9b59b6')
    ax2.bar(x + width/2, results_df['AUC-PR'], width, label='AUC-PR', color='#f39c12')
    ax2.set_xticks(x)
    ax2.set_xticklabels(techniques, rotation=45, ha='right')
    ax2.set_ylabel('AUC Score')
    ax2.set_title('AUC-ROC vs AUC-PR')
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. ROC Curves
    ax3 = axes[1, 0]
    for (name, data), color in zip(detailed_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, data['y_proba'])
        auc = roc_auc_score(y_test, data['y_proba'])
        ax3.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=color, linewidth=2)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Precision-Recall Curves
    ax4 = axes[1, 1]
    for (name, data), color in zip(detailed_results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_test, data['y_proba'])
        ap = average_precision_score(y_test, data['y_proba'])
        ax4.plot(recall, precision, label=f"{name} ({ap:.3f})", color=color, linewidth=2)
    
    # Add baseline (proportion of positives)
    baseline = sum(y_test == 1) / len(y_test)
    ax4.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curves')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resampling_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n Plot saved as 'resampling_comparison.png'")
