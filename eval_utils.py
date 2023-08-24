import numpy as np


class ConfusionMatrix:
    def __init__(self, pred_mask, gt_mask, class_type):
        if class_type == "foreground":
            self.tp = np.sum(np.logical_and(pred_mask, gt_mask))
            self.fp = np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))
            self.fn = np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))
            self.tn = np.sum(np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)))
        elif class_type == "background":
            self.tp = np.sum(np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)))
            self.fp = np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))
            self.fn = np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))
            self.tn = np.sum(np.logical_and(pred_mask, gt_mask))


def iou(pred_mask, gt_mask, class_type="foreground"):
    cm = ConfusionMatrix(pred_mask, gt_mask, class_type)
    return cm.tp / (cm.tp + cm.fp + cm.fn)


def dice(pred_mask, gt_mask, class_type="foreground"):
    cm = ConfusionMatrix(pred_mask, gt_mask, class_type)
    return (2 * cm.tp) / ((2 * cm.tp) + cm.fp + cm.fn)


def cedar(pred_mask, gt_mask, class_type="foreground"):
    assert pred_mask.shape == gt_mask.shape, "pred_mask and gt_mask must have the same shape."
    assert pred_mask.ndim == 2, "pred_mask and gt_mask must be 2D arrays."
    area = pred_mask.shape[0] * pred_mask.shape[1]

    cm = ConfusionMatrix(pred_mask, gt_mask, class_type)
    return dice(pred_mask, gt_mask) - ((cm.fn + cm.fp) / area)


def fig(pred_mask, gt_mask, metric):
    if metric == "iou":
        fg = iou(pred_mask, gt_mask, class_type="foreground")
        bg = iou(pred_mask, gt_mask, class_type="background")
    elif metric == "dice":
        fg = dice(pred_mask, gt_mask, class_type="foreground")
        bg = dice(pred_mask, gt_mask, class_type="background")
    return np.mean([fg, bg])