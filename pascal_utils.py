from PIL import Image
import numpy as np

def get_label_from_num(num):
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse", "motorbike", "person", "plant", "sheep", "sofa", "train", "monitor"]
    return labels[num-1]

def get_queries(img_path):
    img = np.array(Image.open(img_path))

    masks = {}
    uniq_vals = np.unique(img).tolist()

    try:
        uniq_vals.remove(0)
    except:
        print("no 0")
    
    try:
        uniq_vals.remove(255)
    except:
        print("no 255")

    for val in uniq_vals:
        masks[get_label_from_num(val)] = (img == val).astype(int)
            
    return masks