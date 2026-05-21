import numpy as np

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    normalized_mutual_info_score,
    confusion_matrix
)

from sklearn.preprocessing import LabelEncoder

from scipy.optimize import linear_sum_assignment


def calculate_metrics(true_labels, pred_labels):

    true_labels = true_labels.flatten()

    mask = true_labels != 0

    true = true_labels[mask]
    pred = pred_labels[mask]

    nmi = normalized_mutual_info_score(
        true,
        pred
    )

    le_true = LabelEncoder()
    true_enc = le_true.fit_transform(true)

    le_pred = LabelEncoder()
    pred_enc = le_pred.fit_transform(pred)

    cm = confusion_matrix(
        true_enc,
        pred_enc
    )

    row_ind, col_ind = linear_sum_assignment(
        cm,
        maximize=True
    )

    mapping = {}

    for r, c in zip(row_ind, col_ind):
        mapping[c] = r

    pred_mapped = np.array([
        mapping.get(x, x)
        for x in pred_enc
    ])

    acc = accuracy_score(
        true_enc,
        pred_mapped
    )

    kappa = cohen_kappa_score(
        true_enc,
        pred_mapped
    )

    return acc, kappa, nmi, cm