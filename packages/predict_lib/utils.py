import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, classification_report
import math
from scipy.stats import norm

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
    optimal_threshold = thresholds[optimal_idx]
    max_f1_score = f1_scores[optimal_idx]

    # Calculate confusion matrix at the optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    cr = classification_report(y_true, y_pred_optimal, output_dict=True)

    return optimal_threshold, max_f1_score, cr


def implied_prob(S0, K, T, r, sigma):
    """
    计算市场（风险中性）下，价格到期低于或高于执行价的隐含概率
    
    Args:
        S0: 当前标的价格
        K: 执行价
        T: 到期时间（年，5天=5/252）
        r: 无风险利率（小数，如4.1% = 0.041）
        sigma: 隐含波动率（年化，30% = 0.30）
    
    Returns:
        tuple: (prob_below, prob_above) - 跌破执行价的概率和涨破执行价的概率
    """
    if (T <= 0) or (r <= 0) or (sigma <= 0):
        print(f"[Warning] Invalid input. T, r, and sigma must be positive. T={T}, r={r}, sigma={sigma}")
        return 0, 0
    
    d2 = (math.log(S0/K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    prob_below = norm.cdf(-d2)  # P(ST < K)
    prob_above = norm.cdf(d2)   # P(ST > K)
    return prob_below, prob_above