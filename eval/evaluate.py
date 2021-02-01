from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py

from classifier.annotation import annotateTopK
from eval.multilabel_metrics1 import evaluate_per_row_col_metrics

eval_file_image_annotations = '../data/nuswide/nus1_test_annot.txt'
eval_file_scores = '../data/nuswide/caffe-res1-101/neigh2/nus1_test_r101_pred.mat'
topK=3


def _eval_predictions(prediction_scores, prediction_annot, images_annotations, topK):
    print(np.shape(images_annotations))
    print(np.shape(prediction_scores))
    predicted_topK = annotateTopK(prediction_scores, topK)
    resI,resL = evaluate_per_row_col_metrics(images_annotations, prediction_scores, predicted_topK, topK, True)
    per_image_metrics = [resI['prec'],resI['rec'],resI['f1'],resI['map'],resI['acc_top1'],resI['acc_topK']]
    per_label_metrics = [resL['prec'],resL['rec'],resL['f1'],resL['map']]
    print('Per Label Metrics(Prec / Rec / F1 / MAP):')
    print(per_label_metrics)
    print('Per Image Metrics(Prec / Rec / F1 / MAP / Top1 acc / TopK acc):')
    print(per_image_metrics)
    print(np.shape(prediction_annot))
    resI,resL = evaluate_per_row_col_metrics(images_annotations, prediction_scores, prediction_annot, topK, True)
    per_image_metrics = [resI['prec'],resI['rec'],resI['f1'],resI['map'],resI['acc_top1'],resI['acc_topK']]
    per_label_metrics = [resL['prec'],resL['rec'],resL['f1'],resL['map']]
    print('Prob Per Label Metrics(Prec / Rec / F1 / MAP):')
    print(per_label_metrics)
    print('Prob Per Image Metrics(Prec / Rec / F1 / MAP / Top1 acc / TopK acc):')
    print(per_image_metrics)
    return per_label_metrics, per_image_metrics


def _read_files(file_image_annotations,file_prediction_scores):
    with open(file_image_annotations) as fid_image_annotations:
        img_annot_lines = fid_image_annotations.read().splitlines()
    n_imgs = len(img_annot_lines)
    n_labels = len(img_annot_lines[0].split(' '))
    img_annots = np.zeros([n_imgs, n_labels],dtype=np.int32)
    for idx, annot_line in enumerate(img_annot_lines):
        img_annots[idx, :] = [int(val) for val in annot_line.split(' ')]
    scores_db = h5py.File(file_prediction_scores, 'r')
    return img_annots, scores_db['testScores'][:], scores_db['prediction'][:]


def main():
    images_annotations,prediction_scores, prediction_annot = _read_files(eval_file_image_annotations,eval_file_scores)
    _eval_predictions(prediction_scores, prediction_annot, images_annotations, topK)


if __name__ == '__main__':
    main()
