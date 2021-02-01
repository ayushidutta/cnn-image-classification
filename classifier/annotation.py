from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def annotateTopK(predicted_scores, topK):
    n_rows = np.size(predicted_scores, 0)
    n_labels = np.size(predicted_scores, 1)
    predicted_annot = np.zeros([n_rows,n_labels],dtype=np.int32)
    for i in range(n_rows):
        scores_list = list(predicted_scores[i,:])
        for j in range(topK):
            idx = np.argmax(scores_list)
            scores_list[idx] = -float('inf')
            predicted_annot[i, idx] = 1
    return predicted_annot


def annotate_by_probability(predicted_scores, thresh=0.5):
    n_rows = np.size(predicted_scores, 0)
    n_labels = np.size(predicted_scores, 1)
    predicted_annot = np.zeros([n_rows, n_labels], dtype=np.int32)
    for i in range(n_rows):
        gt_idx = np.where(predicted_scores[i, :] > thresh)[0]
        predicted_annot[i, gt_idx] = 1
    return predicted_annot