import torchvision

from attribution_bottleneck.attribution.factory import Factory
from attribution_bottleneck.attribution.per_sample_bottleneck import PerSampleBottleneckReader
from attribution_bottleneck.attribution.readout_bottleneck import ReadoutBottleneckReader
from attribution_bottleneck.bottleneck.readout_bottleneck import \
    OldDenseAdaptiveReadoutBottleneck, DenseAdaptiveReadoutBottleneck
from attribution_bottleneck.utils.data import TorchZooImageNetFolderDataProvider
from attribution_bottleneck.bottleneck.estimator import ReluEstimator, GaussianEstimator, \
    EstimatorGroup
import sys
import torch


def stream_samples(test_set, n_samples=None):
    if n_samples is None:
        n_samples = len(test_set)
    for i in range(n_samples):
        img, target = test_set[i]
        yield img[None], torch.LongTensor([target])


def get_default_config():
    return {
        'seed': 1,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'batch_size': 1,
        'imagenet_train': 'data/imagenet/train',
        'imagenet_test': 'data/imagenet/validation',
        'imagenet_dict_file': 'data/imagenet_label_dict.yaml',
        'imagenet_test_bbox': "data/imagenet/val_bounding_boxes",
        'readout_weighs_vgg16': "weights/readout_vgg16.torch",
        'readout_weighs_resnet50': "weights/readout_resnet50.torch",
        'pattern_weights': 'weights/model_vgg16_pattern_small.pth',
    }


def get_model_and_attribution_method(config):
    model_name = config['model_name']

    data_prov_config = {k: v for k, v in config.items()
                        if k in ['imagenet_train', 'imagenet_test', 'device', 'batch_size',
                                 'imagenet_dict_file']}
    if model_name == "resnet50":

        model = torchvision.models.resnet50(pretrained=True).to(config["device"])
        data_prov = TorchZooImageNetFolderDataProvider(data_prov_config)
        readout_layers = [model.layer1, model.layer2, model.layer3, model.layer4, model.fc]
        readout_types = [*[ReluEstimator for _ in range(4)], GaussianEstimator]
        readout_group = EstimatorGroup(model, [e(l) for e, l in zip(readout_types, readout_layers)])
        gcam_layer = model.layer4

    elif model_name == "vgg16":
        relu = False
        model = torchvision.models.vgg16(pretrained=True).to(config["device"])
        data_prov = TorchZooImageNetFolderDataProvider(data_prov_config)
        readout_feats = [11, 18, 25, 29] if relu else [10, 17, 24, 28]
        readout_layers = [*[model.features[l] for l in readout_feats], model.classifier[-1]]
        readout_types = [*[(ReluEstimator if relu else GaussianEstimator) for _ in readout_feats],
                         GaussianEstimator]
        readout_group = EstimatorGroup(
            model, [e(l) for e, l in zip(readout_types, readout_layers)])
        gcam_layer = model.features[-1]
    else:
        raise

    # Prepare setup
    # Setup(config, model, data_prov)

    # Prepare data
    model = model.to(config["device"])

    print("Model is now on", config["device"])
    sys.stdout.flush()

    # Prepare Readout
    if model_name == "resnet50":
        readout_group.load("pretrained/estim_resnet50/100k_all_layers_fc.torch")
        readout_dense_10 = OldDenseAdaptiveReadoutBottleneck.load_path(
            model, readout_layers, config['readout_weighs_resnet50'])
    elif model_name == "vgg16":
        readout_feats_str = ",".join(str(f) for f in readout_feats)
        readout_path = f"pretrained/estim_vgg16/100k_" + readout_feats_str + ",fc.torch"
        readout_group.load(readout_path)
        readout_dense_10 = DenseAdaptiveReadoutBottleneck.load_path(
            model, readout_layers, config['readout_weighs_vgg16'])

    lit = Factory(model)

    attribution = {
        'Fitted': lambda: PerSampleBottleneckReader(model, readout_group.estimators[1]),
        'Fitted beta1': lambda: PerSampleBottleneckReader(
            model, readout_group.estimators[1], beta=1),
        'Fitted beta100': lambda: PerSampleBottleneckReader(
            model, readout_group.estimators[1], beta=100),
        'Readout Dense 10': lambda: ReadoutBottleneckReader(
            model, readout_layers[0], readout_dense_10),
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
    }[config['attribution_name']]()

    test_set = data_prov.data_fac.test_set()
    return model, attribution, test_set
