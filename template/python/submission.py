from typing import Dict, List, Tuple
from collections import Counter
from itertools import combinations
import math

class Solution:

    def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
        
        Args:
          true_labels: list of true labels
          pred_labels: list of predicted labels
        
        Returns:
          A dictionary where keys are tuples (true_label, pred_label), and values are the counts of occurrences.
        """
        # Initialize an empty dictionary to store counts
        matrix = {}
        
        # Loop through each true and predicted label pair
        for t, p in zip(true_labels, pred_labels):
            # Count occurrences of each (true_label, pred_label) pair
            matrix[(t, p)] = matrix.get((t, p), 0) + 1
            
        return matrix

    def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """
        Calculate the Jaccard index for clustering validation.
        
        The Jaccard index measures the similarity between two sets of clusters by comparing the intersection
        and union of points assigned to the same clusters in both true and predicted labels.
        
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        
        Returns:
          The Jaccard index. Do NOT round this value.
        """
        # Helper function to count the number of pairs within each cluster in the list
        # This counts how many pairs of points are in the same cluster in the list of labels
        def count_pairs(labels: List[int]) -> int:
            label_counts = Counter(labels)
            # For each cluster, count pairs of points that can be formed
            return sum(count * (count - 1) // 2 for count in label_counts.values())
        
        # Calculate intersection: pairs where both true and predicted labels place points in the same cluster
        # We count all unique pairs of indices (i, j) that satisfy this condition
        intersection = sum(
            1 for (i, j) in combinations(range(len(true_labels)), 2)
            if (true_labels[i] == true_labels[j] and pred_labels[i] == pred_labels[j])
        )

        # Count pairs in true labels and predicted labels
        true_pairs = count_pairs(true_labels)
        pred_pairs = count_pairs(pred_labels)
        
        # Union is calculated using inclusion-exclusion: true_pairs + pred_pairs - intersection
        union = true_pairs + pred_pairs - intersection
        
        # Jaccard index is the ratio of intersection to union
        return intersection / union if union != 0 else 0.0
        
    def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """
        Calculate the Normalized Mutual Information (NMI) between true and predicted labels.
        
        NMI is a measure of the amount of shared information between two clustering assignments.
        It scales mutual information by the entropies of the true and predicted labels.
        
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        
        Returns:
          The Normalized Mutual Information. Do NOT round this value.
        """
        # Total number of data points
        n = len(true_labels)
        
        # Count occurrences of each unique label in true and predicted lists
        true_counter = Counter(true_labels)
        pred_counter = Counter(pred_labels)
        
        # Count occurrences of each (true_label, pred_label) pair (joint distribution)
        joint_counter = Counter(zip(true_labels, pred_labels))

        # Calculate Mutual Information (MI)
        mi = 0.0  # Initialize MI to zero
        for (t, p), count in joint_counter.items():
            # Probability of true label t
            p_true = true_counter[t] / n
            # Probability of predicted label p
            p_pred = pred_counter[p] / n
            # Joint probability of (true label, predicted label) pair
            p_joint = count / n
            # Increment MI by the contribution of this pair
            mi += p_joint * math.log(p_joint / (p_true * p_pred), 2)

        # Calculate entropy of true labels
        h_true = -sum((count / n) * math.log(count / n, 2) for count in true_counter.values())
        
        # Calculate entropy of predicted labels
        h_pred = -sum((count / n) * math.log(count / n, 2) for count in pred_counter.values())

        # Calculate Normalized Mutual Information (NMI) using the entropies and mutual information
        # NMI = MI / sqrt(H(true) * H(pred))
        nmi = mi / math.sqrt(h_true * h_pred) if h_true > 0 and h_pred > 0 else 0.0
        return nmi
