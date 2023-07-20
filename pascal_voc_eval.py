import cv2
import numpy as np

from pascal_utils import get_queries
from grounded_sam import run_grounded_sam, env_setup, load_models


def get_iou(pred_mask, gt_mask):
    # # Ensure the masks are binary
    # pred_mask = pred_mask.astype(np.bool)
    # gt_mask = gt_mask.astype(np.bool)

    # Compute Intersection
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def pascal_voc_eval():
    # Specify path 
    val_id_path = 'datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    class_name_path = 'datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/class_names.txt'
    img_folder_path = 'datasets/pascal/VOCdevkit/VOC2012/JPEGImages'
    gt_folder_path = 'datasets/pascal/VOCdevkit/VOC2012/SegmentationClass'

    # Load class names
    class_names = []
    for line in open(class_name_path, 'r'):
        class_names.append(line.strip())
    
    # Load val ids
    val_ids = []
    for line in open(val_id_path, 'r'):
        id = line.strip()
        val_ids.append(id)

    # Load model
    env_setup()
    groundingdino_model, sam_predictor = load_models()

    # Evaluation
    ctr = 0
    iou_sum = 0
    for id in val_ids:
        # load image
        img_path = img_folder_path + '/' + id + '.jpg'

        # load ground truth
        gt_path = gt_folder_path + '/' + id + '.png'
        gt_masks = get_queries(gt_path) # to be implemented by vignav -> ["text_prompt": "aeroplane", "gt_mask": np_array]

        # run through all classes
        for class_name in class_names:
            if class_name not in gt_masks:
                continue

            # load gt mask
            gt_mask = gt_masks[class_name]

            # run grounded sam
            text_prompt = 'a ' + class_name
            pred_mask = run_grounded_sam(img_path, text_prompt, groundingdino_model, sam_predictor) # to be implemented by vignav -> np_array
            pred_mask = (pred_mask != 0).astype(int)

            # compute iou
            iou_score = get_iou(pred_mask, gt_mask)
            print(id, class_name, iou_score)
            iou_sum += iou_score
            ctr += 1
    mIoU = iou_sum / ctr
    return mIoU
        

if __name__ == '__main__':
    mIoU = pascal_voc_eval()
    print(mIoU)