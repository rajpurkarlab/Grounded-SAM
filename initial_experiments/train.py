"""Training scripts for Grounded SAM.

Iteratively train between adaptation and segmentation objectives:
    - Adaptation: train Grounded SAM to align with frozen biomed CLIP using medical dataset.
    - Segmentation: train Grounded SAM with its original objective using natural dataset.
"""