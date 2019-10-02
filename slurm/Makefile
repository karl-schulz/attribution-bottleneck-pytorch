SHELL = bash
.ONESHELL:

eval_resnet50_8x8:
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Fitted"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Gradient"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Saliency"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Guided Backpropagation"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Grad-CAM"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Occlusion-14x14"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Occlusion-8x8"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Random"

test_deg_resnet50_8x8:
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Readout Dense 10" "test"

eval_readout:
	# sensitivity
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Readout Dense 10"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Readout Dense 10"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Readout Dense 10"
	# degradation
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Readout Dense 10"
	# bbox
	sbatch ./slurm_eval_bbox.sh "resnet50" "Readout Dense 10"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Readout Dense 10"

test_readout_vgg16:
	sbatch ./slurm_eval_degradation.sh "vgg16" "Readout Dense 10" "test"

eval_readout_vgg16:
	# sensitivity
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Readout Dense 10"
	# degradation
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Readout Dense 10"
	# bbox
	sbatch ./slurm_eval_bbox.sh "vgg16" "Readout Dense 10"

eval_readout_resnet50:
	# sensitivity
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Readout Dense 10"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Readout Dense 10"
	# degradation
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Readout Dense 10"
	# bbox
	sbatch ./slurm_eval_bbox.sh "resnet50" "Readout Dense 10"

eval_lrp:
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "LRP"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "LRP"

eval_deg_fitted_beta:
	#sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Fitted beta1"
	#sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Fitted beta100"
	#sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Fitted beta1"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Fitted beta100"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Fitted beta1"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Fitted beta100"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Fitted beta1"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Fitted beta100"

eval_resnet50_14x14:
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Fitted"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Gradient"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Saliency"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Guided Backpropagation"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Grad-CAM"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Occlusion-14x14"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Occlusion-8x8"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "Random"

eval_vgg_8x8:
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Fitted"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Gradient"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Saliency"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Guided Backpropagation"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Grad-CAM"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "GuidedGrad-CAM"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Occlusion-8x8"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Occlusion-14x14"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "Random"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "PatternAttribution"

eval_vgg_14x14:
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Fitted"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Readout Dense 10"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Gradient"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Saliency"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Guided Backpropagation"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Grad-CAM"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "GuidedGrad-CAM"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Occlusion-14x14"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Occlusion-8x8"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "Random"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "PatternAttribution"

test_eval_resnet50:
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "Fitted"

eval_guidedgradcam:
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "GuidedGrad-CAM"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "GuidedGrad-CAM"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1  "GuidedGrad-CAM"
	sbatch ./slurm_eval_degradation.sh "resnet50" 8 "GuidedGrad-CAM"
	sbatch ./slurm_eval_degradation.sh "resnet50" 14 "GuidedGrad-CAM"
	sbatch ./slurm_eval_degradation.sh "vgg16" 8 "GuidedGrad-CAM"
	sbatch ./slurm_eval_degradation.sh "vgg16" 14 "GuidedGrad-CAM"
	# sbatch ./slurm_eval_bbox.sh "resnet50" "GuidedGrad-CAM"
	# sbatch ./slurm_eval_bbox.sh "vgg16" "GuidedGrad-CAM"


eval_bbox_resnet50:
	sbatch ./slurm_eval_bbox.sh "resnet50" "Fitted"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Readout Dense 10"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Gradient"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Saliency"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Guided Backpropagation"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Grad-CAM"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Occlusion-14x14"
	sbatch ./slurm_eval_bbox.sh "resnet50" "Occlusion-8x8"


eval_bbox_vgg16:
	sbatch ./slurm_eval_bbox.sh "vgg16" "Fitted"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Fitted beta1"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Fitted beta100"
	# sbatch ./slurm_eval_bbox.sh "vgg16" "Readout Dense 10"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Gradient"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Saliency"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Guided Backpropagation"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Grad-CAM"
	sbatch ./slurm_eval_bbox.sh "vgg16" "GuidedGrad-CAM"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Occlusion-14x14"
	sbatch ./slurm_eval_bbox.sh "vgg16" "Occlusion-8x8"
	sbatch ./slurm_eval_bbox.sh "vgg16" "LRP"
	sbatch ./slurm_eval_bbox.sh "vgg16" "PatternAttribution"

sensitivity_n_resnet50_tile1:
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Fitted"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Fitted beta1"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Fitted beta100"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Readout Dense 10"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Gradient"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Saliency"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Guided Backpropagation"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Grad-CAM"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Occlusion-14x14"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Occlusion-8x8"

sensitivity_n_resnet50_tile8:
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Fitted"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Fitted beta1"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Fitted beta100"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Readout Dense 10"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Gradient"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Saliency"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Guided Backpropagation"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Grad-CAM"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Occlusion-14x14"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Occlusion-8x8"

sensitivity_n_vgg16_tile1:
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Fitted"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Fitted beta1"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Fitted beta100"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Readout Dense 10"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Gradient"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Saliency"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Guided Backpropagation"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "GuidedGrad-CAM"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Occlusion-14x14"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Occlusion-8x8"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "LRP"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "PatternAttribution"

sensitivity_n_vgg16_tile8:
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Fitted"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Fitted beta1"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Fitted beta100"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Readout Dense 10"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Gradient"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Saliency"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Smoothgrad (of Saliency)"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Int. Grad. (of Saliency)"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Guided Backpropagation"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Grad-CAM"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "GuidedGrad-CAM"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Occlusion-14x14"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "Occlusion-8x8"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "LRP"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 8 "PatternAttribution"

test_sensitivity_n:
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Guided Backpropagation" "test"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Occlusion-8x8"  "test"

test_sensitivity_resnet:
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Readout Dense 10" "test"

sensitivity_n_occlusion:
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Occlusion-8x8"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Occlusion-8x8"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Occlusion-8x8"
	sbatch ./slurm_eval_sensitivity_n.sh "vgg16" 1 "Occlusion-14x14"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 1 "Occlusion-14x14"
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Occlusion-14x14"

test_occlusion_sensitivity:
	sbatch ./slurm_eval_sensitivity_n.sh "resnet50" 8 "Occlusion-8x8" "test"

test_bbox_tasks="test-bbox-tasks.txt"
test_bbox_jobs = $(shell cat $(test_bbox_tasks) | wc -l )
test_bbox_jobarray:
	@echo "running job array with $(test_bbox_jobs) jobs:"
	@cat $(test_bbox_tasks)
	sbatch --array=1-$(test_bbox_jobs)%6 ./slurm_eval_bbox.sh $(test_bbox_tasks)

bbox_tasks="bbox-tasks.txt"
bbox_jobs = $(shell cat $(bbox_tasks) | wc -l )
bbox_jobarray:
	@echo "running job array with $(bbox_jobs) jobs:"
	@cat $(bbox_tasks)
	sbatch --array=1-$(bbox_jobs)%6 ./slurm_eval_bbox.sh $(bbox_tasks)

