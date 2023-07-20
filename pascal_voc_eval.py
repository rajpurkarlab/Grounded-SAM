import cv2
import numpy as np
import json

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
    iou_results = {class_name: [] for class_name in class_names}
    for id in val_ids:
        # load image
        img_path = img_folder_path + '/' + id + '.jpg'

        # load ground truth
        gt_path = gt_folder_path + '/' + id + '.png'
        gt_masks = get_queries(gt_path)

        # run through all classes
        for class_name in class_names:
            if class_name not in gt_masks:
                continue

            # load gt mask
            gt_mask = gt_masks[class_name]

            # run grounded sam
            text_prompt = 'a ' + class_name

            try:
                pred_mask = run_grounded_sam(img_path, text_prompt, groundingdino_model, sam_predictor)
                pred_mask = (pred_mask != 0).astype(int)

                # compute iou
                iou_score = get_iou(pred_mask, gt_mask)
                print(id, class_name, iou_score) 
                
                iou_results[class_name].append(iou_score)
            except:
                print(f"\nSkipping {img_path}, {text_prompt} due to errors\n")

    # compute average mIoU across all classes
    total_sum = 0
    total_count = 0
    for value_list in iou_results.values():
        total_sum += sum(value_list)
        total_count += len(value_list)
    mIoU = total_sum / total_count

    # compute mIoU by class and save to json
    mIoU_classes = {}
    for class_name in class_names:
        mIoU_classes[class_name] = np.mean(iou_results[class_name])
    mIoU_classes['mIoU'] = mIoU
    json.dump(mIoU_classes, open('pascal_miou.json', 'w'))
    
    return mIoU
        

if __name__ == '__main__':
    mIoU = pascal_voc_eval()
    print(mIoU)