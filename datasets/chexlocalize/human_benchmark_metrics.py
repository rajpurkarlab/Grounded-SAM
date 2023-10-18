"""
Calculate human benchmark metrics on CheXlocalize test set.

There are 668 chest X-rays from 500 patients in the CheXlocalize test set.

We remove the CXR associated with id `patient65086_study1_view1_frontal` because
there are no ground-truth segmentations recorded even though there are positive
labels for this CXR.

We remove the CXR associated with id `patient65145_study1_view1_frontal` because
there are no human benchmark segmentations even though there are ground-truth
segmentations. Since human benchmark radiologists were given the ground-truth
labels, we suspect that it is an error that it's missing from hb_dict.

We remove the CXRs associated with ids `patient64860_study1_view1_frontal`,
`patient64860_study1_view2_lateral`, and `patient65192_study1_view1_frontal`
because there are no human benchmark segmentations for them even though their
ids are included in `hb_dict`.

Therefore, we have:
- 666 CXRs in the test set
- 498 CXRs with ground-truth segmentations
- 498 CXRs with human benchmark segmentations
"""

from argparse import ArgumentParser
import json
from statistics import mean

import numpy as np
import pandas as pd
from pycocotools import mask

from metrics import balanced_metric


TASKS = ["Airspace Opacity",
         "Atelectasis",
         "Cardiomegaly",
         "Consolidation",
         "Edema",
         "Enlarged Cardiomediastinum",
         "Lung Lesion",
         "Pleural Effusion",
         "Pneumothorax",
         "Support Devices"]


def seg_dict_full_test_set(seg_dict, labels):
    """
    `seg_dict` only includes those CXRs with at least one positive ground-truth
    label, but each CXR id key has values for all ten pathologies (regardless of
    ground-truth label).

    This script adds all CXRs to `seg_dict`, and for each CXR, it adds a key
    `labels` with the ground-truth labels for each of the ten pathologies.
    """
    for id in labels.cxr_id:
        id_labels = labels[labels.cxr_id == id]

        labels_dict = {}
        for task in TASKS:
            labels_dict[task] = int(id_labels[task].item())

        if id in seg_dict:
            seg_dict[id]['labels'] = labels_dict
        else:
            seg_dict[id] = {'labels': labels_dict}

    return seg_dict


def calculate_metrics(gt_dict, hb_dict):
    assert sorted(gt_dict.keys()) == sorted(hb_dict.keys())
    cxr_ids = sorted(gt_dict.keys())

    metrics = {"task": [], "balanced_iou": [], "balanced_dice": []}
    for task in TASKS:
        print(f"Evaluating {task}...")

        balanced_iou = []
        balanced_dice = []
        for cxr_id in cxr_ids:
            label = gt_dict[cxr_id]['labels'][task]

            # get ground-truth segmentation mask
            if task in gt_dict[cxr_id]:
                gt_mask = mask.decode(gt_dict[cxr_id][task])

            # get human benchmark segmentation mask
            if task in hb_dict[cxr_id]:
                hb_mask = mask.decode(hb_dict[cxr_id][task])

            if label == 0:
                if task in hb_dict[cxr_id]:
                    if np.sum(hb_mask) == 0:
                        balanced_iou.append(1.0)
                        balanced_dice.append(1.0)
                    else:
                        if task in gt_dict[cxr_id]:
                            assert np.sum(gt_mask) == 0, "gt_mask should be all zeros"
                        else:
                            gt_mask = np.zeros(hb_dict[cxr_id][task]['size'])
                        balanced_iou.append(balanced_metric(hb_mask, gt_mask, metric="iou"))
                        balanced_dice.append(balanced_metric(hb_mask, gt_mask, metric="dice"))
                else:
                    if task in gt_dict[cxr_id]:
                        assert np.sum(gt_mask) == 0, "gt_mask should be all zeros"
                    balanced_iou.append(1.0)
                    balanced_dice.append(1.0)
            else:
                # if label == 1, gt_mask and hb_mask should both exist
                assert gt_mask.shape == hb_mask.shape
                balanced_iou.append(balanced_metric(hb_mask, gt_mask, metric="iou"))
                balanced_dice.append(balanced_metric(hb_mask, gt_mask, metric="dice"))
        metrics["task"].append(task)
        metrics["balanced_iou"].append(mean(balanced_iou))
        metrics["balanced_dice"].append(mean(balanced_dice))
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str,
                        help='json path where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--hb_path', type=str,
                        help='json path where human benchmark segmentations are \
                            saved (encoded)')
    parser.add_argument('--gt_labels_path', type=str,
                        help='csv path where ground-truth labels are saved')
    args = parser.parse_args()

    with open(args.gt_path) as f:
        gt_dict = json.load(f)
    with open(args.hb_path) as f:
        hb_dict = json.load(f)
    labels = pd.read_csv(args.gt_labels_path)
    labels = labels.rename(columns={"Lung Opacity": "Airspace Opacity"})
    labels["cxr_id"] = labels["Path"].apply(
        lambda x: "_".join(x.split(".")[0].split("/")[1:]))

    # remove patients per the script's comment above
    for id in ["patient65086_study1_view1_frontal", "patient65145_study1_view1_frontal"]:
        if id in gt_dict:
            gt_dict.pop(id)
        if id in hb_dict:
            hb_dict.pop(id)
        labels = labels.drop(labels[labels.cxr_id == id].index)
    for id in ["patient64860_study1_view1_frontal",
               "patient64860_study1_view2_lateral",
               "patient65192_study1_view1_frontal"]:
        hb_dict.pop(id)

    # add all CXR ids to gt_dict and hb_dict
    gt_dict = seg_dict_full_test_set(gt_dict, labels)
    hb_dict = seg_dict_full_test_set(hb_dict, labels)

    # calculate metrics
    metrics = calculate_metrics(gt_dict, hb_dict)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("human_benchmark_metrics.csv", index=False)