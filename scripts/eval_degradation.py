#!/usr/bin/env python
# coding: utf-8

import sys
import torchvision
from attribution.literature import *
from attribution.info_empirical import *
from attribution.info_bottleneck import *
from attribution.info_readout import *
from train.setup import *
from utils.data import *
from utils.misc import *
from information.readout_bottleneck import *
from information.fitted_bottleneck import *
from information.estimator import *
from models.allcnn96 import *
from evaluate.tensorview import *
from evaluate.degradation import *
from evaluate.visual import *

torch.backends.cudnn.benchmark = True

try:
    testing = (sys.argv[4] == 'test')
except IndexError:
    testing = False


if testing:
    print("testing run. reducing samples to 50!")
    n_samples = 50
else:
    n_samples = 50000

config = {
    'seed': 1,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'batch_size': 3,
    'n_samples': n_samples,
    'imagenet_train': 'data/imagenet/train',
    'imagenet_test': 'data/imagenet/validation',
    'imagenet_dict_file': 'data/imagenet/imagenet_label_dict.yaml',
    'readout_weighs_vgg16':
        "pretrained/readout_vgg16/09-14_11-32-21_VGG_e015_b00000_s19217.torch",
    'readout_weighs_resnet50':
        "pretrained/readout_resnet50/09-14_11-32-47_ResNet_e009_b40000_s12410.torch"
}


model_name = sys.argv[1]
patch_size = int(sys.argv[2])
attribution_name = sys.argv[3]
dev = torch.device(config['device'])
print("Evaluation {} on model {} with patch size {}x{}:".format(attribution_name, model_name, patch_size, patch_size))
print("config is:", config)


if model_name == "resnet50":
    pattern = None
    model = torchvision.models.resnet50(pretrained=True).to(config["device"])
    data_prov = TorchZooImageNetFolderDataProvider(config)
    readout_layers = [model.layer1, model.layer2, model.layer3, model.layer4, model.fc]
    readout_types = [*[ReluEstimator for _ in range(4)], GaussianEstimator]
    readout_group = EstimatorGroup(model, [e(l) for e, l in zip(readout_types, readout_layers)])
    fitted_estim = readout_group.estimators[1]
    gcam_layer = model.layer4

elif model_name == "vgg16":
    relu = False
    # our vgg weights do not need a special transformation
    pattern_transform = False
    pattern = PatternWeightsProvider(config["device"])
    model = pattern.model
    data_prov = TorchZooImageNetFolderDataProvider(config, transform=PatternTransform() if pattern_transform else None)
    readout_feats = [11,18,25,29] if relu else [10,17,24,28]
    readout_layers = [*[model.features[l] for l in readout_feats], model.classifier[-1]]
    readout_types = [*[(ReluEstimator if relu else GaussianEstimator) for _ in readout_feats], GaussianEstimator]
    readout_group = EstimatorGroup(model, [e(l) for e, l in zip(readout_types, readout_layers)])
    fitted_estim = readout_group.estimators[1]
    gcam_layer = model.features[-1]

else:
    raise

# Prepare setup
setup = Setup(config, model, data_prov)
empirical_group = EstimatorGroup(model, readout_group.estimators[2:-1])

# Prepare data
img_dim = 224 if model_name != "allcnn96" else 96
model = model.to(config["device"])
print("Model is now on", config["device"])



# Prepare Readout
if model_name == "resnet50":
    readout_group.load("pretrained/estim_resnet50/100k_all_layers_fc.torch")
    readout_dense_10 = OldDenseAdaptiveReadoutBottleneck.load_path(
        model, readout_layers, config['readout_weighs_resnet50'])
elif model_name == "vgg16":
    ltype = "relu" if relu else "conv"
    ptype = "_pattern" if pattern_transform else ""
    readout_path = f"pretrained/estim{ptype}_vgg16/100k_"+",".join(str(f) for f in readout_feats)+",fc.torch"
    readout_group.load(readout_path)
    readout_dense_10 = DenseAdaptiveReadoutBottleneck.load_path(
        model, readout_layers, config['readout_weighs_vgg16'])




lit = Literature(model, pattern)
t = patch_size
methods = {
    'Fitted': lambda: FittedBottleneckReader(model, readout_group.estimators[1]),
    'Fitted beta1': lambda: FittedBottleneckReader(model, readout_group.estimators[1], beta=1),
    'Fitted beta100': lambda: FittedBottleneckReader(model, readout_group.estimators[1], beta=100),
    'Readout Dense 10': lambda: ReadoutBottleneckReader(model, readout_layers[0], readout_dense_10),
    'Gradient': lambda: lit.Gradient(),
    'Saliency': lambda: lit.Saliency(),
    'Smoothgrad (of Saliency)': lambda: lit.SmoothGrad(),
    'Int. Grad. (of Saliency)': lambda: lit.IntegratedGradients(),
    'Guided Backpropagation': lambda: lit.GuidedBackprop(),
    'Grad-CAM': lambda: lit.GradCAM(gcam_layer),
    'Occlusion-14x14': lambda: lit.Occlusion(patch_size=14),
    'Occlusion-8x8': lambda: lit.Occlusion(patch_size=8),
    'Random': lambda: lit.Random(),
    'PatternAttribution': lambda: lit.PatternAttribution(),
    'LRP': lambda: lit.LRP()
}

method = methods[attribution_name]()

model.eval()

evaluations = {
    f"{t}{t}": DegradationEval(model, tile_size=(t,t)),
    f"{t}{t} reversed": DegradationEval(model, tile_size=(t,t), reverse=True),
}

result_list = []

for name, ev in evaluations.items():
    collector = Collector(ev, {attribution_name: method})
    data_gen = data_prov.gen_samples(config['n_samples'], "cpu", test_set=True)
    result_list.append(collector.make_eval(data_gen, config['n_samples']))


time = strftime("%m-%d_%H-%M-%S", gmtime())
if testing:
    fname = f"results/test_{model_name}_{attribution_name}_{t}x{t}_{time}.torch"
else:
    fname = f"results/{model_name}_{attribution_name}_{t}x{t}_{time}.torch"
torch.save(result_list, fname)
print("Saved:", fname)
