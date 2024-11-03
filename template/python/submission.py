# Submit this file to Gradescope
from typing import Dict, List, Tuple
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.
from collections import Counter
from itertools import combinations
import math

class Solution:

    def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
        """Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
        Args:
          true_labels: list of true labels
          pred_labels: list of predicted labels
        Returns:
          A dictionary of (true_label, pred_label): count
        """
        matrix = {}
        for t, p in zip(true_labels, pred_labels):
            matrix[(t, p)] = matrix.get((t, p), 0) + 1
        return matrix

    def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """Calculate the Jaccard index.
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        Returns:
          The Jaccard index. Do NOT round this value.
        """
        # Count pairs of points that are in the same cluster
        def count_pairs(labels: List[int]) -> int:
            label_counts = Counter(labels)
            return sum(count * (count - 1) // 2 for count in label_counts.values())
        
        # Calculate the intersection: pairs with same true and predicted clusters
        intersection = sum(
            1 for (i, j) in combinations(range(len(true_labels)), 2)
            if (true_labels[i] == true_labels[j] and pred_labels[i] == pred_labels[j])
        )

        # Calculate the union of pairs
        true_pairs = count_pairs(true_labels)
        pred_pairs = count_pairs(pred_labels)
        union = true_pairs + pred_pairs - intersection
        
        return intersection / union if union != 0 else 0.0
        
    def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """Calculate the normalized mutual information.
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        Returns:
          The normalized mutual information. Do NOT round this value.
        """
        # Compute label frequencies
        n = len(true_labels)
        true_counter = Counter(true_labels)
        pred_counter = Counter(pred_labels)
        joint_counter = Counter(zip(true_labels, pred_labels))

        # Calculate mutual information
        mi = 0.0
        for (t, p), count in joint_counter.items():
            p_true = true_counter[t] / n
            p_pred = pred_counter[p] / n
            p_joint = count / n
            mi += p_joint * math.log(p_joint / (p_true * p_pred), 2)

        # Calculate entropies
        h_true = -sum((count / n) * math.log(count / n, 2) for count in true_counter.values())
        h_pred = -sum((count / n) * math.log(count / n, 2) for count in pred_counter.values())

        # Normalized Mutual Information
        nmi = mi / math.sqrt(h_true * h_pred) if h_true > 0 and h_pred > 0 else 0.0
        return nmi
