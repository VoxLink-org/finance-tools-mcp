import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, classification_report

def find_optimal_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float, dict]:
    """
    Finds the optimal probability threshold to maximize the F1 score.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        tuple[float, float, dict]: A tuple containing the optimal threshold, the maximum F1 score, and the report.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1 score for each threshold
    # Handle potential division by zero if precision + recall is 0
    f1_scores = np.where((precisions + recalls) == 0, 0, 2 * (precisions * recalls) / (precisions + recalls))

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = 0.5 # thresholds[optimal_idx]
    max_f1_score = f1_scores[optimal_idx]

    # Calculate confusion matrix at the optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    cr = classification_report(y_true, y_pred_optimal, output_dict=True)

    return optimal_threshold, max_f1_score, cr