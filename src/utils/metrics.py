import numpy as np

def levenshtein(a, b):
    dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (a[i-1] != b[j-1])
            )
    return dp[len(a)][len(b)]

def compute_per(pred_list, true_list):
    """
    Compute Phoneme Error Rate.
    pred_list: List of lists of phoneme IDs (or strings)
    true_list: List of lists of phoneme IDs (or strings)
    """
    total = 0
    dist = 0
    for p, t in zip(pred_list, true_list):
        dist += levenshtein(p, t)
        total += max(len(t), 1)
    return dist / total if total > 0 else 1.0

def compute_accuracy(per):
    return max(0, (1 - per) * 100)
