# Grounded-SAM

## Preliminary Works

Below is background knowledge on the models used in our experiments.

<div align="center">

| Title | Intro | Description | Links |
|:----:|:----:|:----:|:----:|
| [Segment-Anything](https://arxiv.org/abs/2304.02643) | ![](https://github.com/facebookresearch/segment-anything/blob/main/assets/model_diagram.png?raw=true) | A strong foundation model aims to segment everything in an image, which needs prompts (as boxes/points/text) to generate masks | [[Github](https://github.com/facebookresearch/segment-anything)] <br> [[Page](https://segment-anything.com/)] <br> [[Demo](https://segment-anything.com/demo)] |
| [Grounding DINO](https://arxiv.org/abs/2303.05499) | ![](https://github.com/IDEA-Research/GroundingDINO/blob/main/.asset/hero_figure.png?raw=True) | A strong zero-shot detector which is capable of to generate high quality boxes and labels with free-form text. | [[Github](https://github.com/IDEA-Research/GroundingDINO)] <br> [[Demo](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo)] | 

</div>

## Installation
Simply run:
```
pip install -r requirements.txt
cd models && python3 install.py
```
If using O2, first run:
```
module load python/3.10.11 gcc/9.2.0 cuda/11.7
```