# The Attribution Bottleneck

This the source code for the paper "TODO: ADD TITLE". The main point
of this repository is to reproduce our results.

## Setup

1.Create a conda environment with all packages:
```bash
$ conda create -n new environment --file requirements.txt
```
2. Download the model weights from the [release page](releases)

## Usage

We provide


## Scripts

The scripts to reproduce our evaluation can be found in the [scripts
directory](scripts).
Following attributions are implemented:



For the bounding box task, replace the model with either `vgg16` or `resnet50`.
```bash
$eval_bounding_boxes.py [model] [attribution]
```

For the degradation task, you also have specify the tile size. In the paper, we
used `8` and `14`.
```bash
$ eval_degradation.py [model] [tile size] [attribution]
```

The results on sensitivity-n can be calculated with:
```bash
eval_sensitivity_n.py [model] [tile size] [attribution]
```

