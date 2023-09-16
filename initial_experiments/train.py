"""Training scripts for Grounded SAM.

Iteratively train between adaptation and segmentation objectives:
    - Adaptation: train Grounded SAM to align with frozen biomed CLIP using medical dataset.
    - Segmentation: train Grounded SAM with its original objective using natural dataset.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image

from linear_probe import LinearProbe
from adaptation_loss import load_models, compute_loss
