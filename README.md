# The Attribution Bottleneck

This the source code for the paper "TODO: ADD TITLE". The main point
of this repository is to reproduce our results.

## Setup

1.Create a conda environment with all packages:
```bash
$ conda create -n new environment --file requirements.txt
```
2. Using your new conda environment, install this repository with pip: `pip install .`
3. Download the model weights from the [release page](releases) and unpack them
   in the repository root directory: `tar -xvf bottleneck_for_attribution_weights.tar.gz`.
4. Place the imagenet dataset under `data/imagenet`. You might just create
   a link with `ln -s [image dir]  data/imagenet`.
5. Test it with: `python ./scripts/eval_degradation.py resnet50 8 Saliency test`

## Usage

We provide some jupyter notebooks.....


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

