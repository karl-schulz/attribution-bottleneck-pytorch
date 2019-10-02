#!/usr/bin/env python
# coding: utf-8

import sys
import os
import torch
from attribution_bottleneck.evaluate.script_utils import stream_samples, \
    get_model_and_attribution_method, get_default_config
from attribution_bottleneck.evaluate.degradation import DegradationEval, run_evaluation
from time import strftime, gmtime
torch.backends.cudnn.benchmark = True

try:
    testing = (sys.argv[4] == 'test')
except IndexError:
    testing = False


if testing:
    print("testing run. reducing samples to 50!")
    n_samples = 1
else:
    n_samples = 50000

model_name = sys.argv[1]
patch_size = int(sys.argv[2])
attribution_name = sys.argv[3]

config = get_default_config()
config.update({
    'result_dir': 'results/deg/' + str(patch_size),
    'model_name': model_name,
    'attribution_name': attribution_name,
    'n_samples': n_samples,
})


dev = torch.device(config['device'])
print("Evaluation {} on model {} with patch size {}x{}:".format(attribution_name,
                                                                model_name, patch_size, patch_size))
print("config is:", config)


model, attribution, test_set = get_model_and_attribution_method(config)
model.eval()

t = patch_size


morf_result, morf_time = run_evaluation(
    DegradationEval(model, tile_size=(t, t)), attribution,
    stream_samples(test_set, config['n_samples']), config['n_samples'])

lerf_result, lerf_time = run_evaluation(
    DegradationEval(model, tile_size=(t, t), reverse=True), attribution,
    stream_samples(test_set, config['n_samples']), config['n_samples'])

time = strftime("%m-%d_%H-%M-%S", gmtime())

result_dir = config['result_dir']
if "SLURM_ARRAY_JOB_ID" in os.environ:
    result_dir = os.path.join(result_dir, os.environ['SLURM_ARRAY_JOB_ID'])
os.makedirs(result_dir, exist_ok=True)

if testing:
    fname = f"test_{model_name}_{attribution_name}_{t}x{t}_{time}.torch"
else:
    fname = f"{model_name}_{attribution_name}_{t}x{t}_{time}.torch"

output_filename = os.path.join(result_dir, fname)


torch.save({
    'config': config,
    'result': {
        'morf': morf_result,
        'lerf': lerf_result,
        'lerf_time': lerf_time,
        'morf_time': morf_time,
    }
}, output_filename)

print("Saved:", output_filename)
