SHELL = bash



test_bbox_tasks="test-tasks-bbox.txt"
test_bbox_jobs = $(shell cat $(test_bbox_tasks) | wc -l )
test_bbox_jobarray:
	@echo "running job array with $(test_bbox_jobs) jobs:"
	@cat $(test_bbox_tasks)
	sbatch --array=1-$(test_bbox_jobs) ./slurm_eval_bbox.sh $(test_bbox_tasks)

bbox_tasks="tasks-bbox.txt"
bbox_jobs = $(shell cat $(bbox_tasks) | wc -l )
bbox_jobarray:
	@echo "running job array with $(bbox_jobs) jobs:"
	@cat $(bbox_tasks)
	sbatch --array=1-$(bbox_jobs) ./slurm_eval_bbox.sh $(bbox_tasks)



test_deg_tasks="test-tasks-degradation-8.txt"
test_deg_jobs = $(shell cat $(test_deg_tasks) | wc -l )
test_deg_jobarray:
	@echo "running job array with $(test_deg_jobs) jobs:"
	@cat $(test_deg_tasks)
	sbatch --array=1-$(test_deg_jobs) ./slurm_eval_degradation.sh $(test_deg_tasks)

deg8_tasks="tasks-degradation-8.txt"
deg8_jobs = $(shell cat $(deg8_tasks) | wc -l )
deg8_jobarray:
	@echo "running job array with $(deg_jobs) jobs:"
	@cat $(deg8_tasks)
	sbatch --array=1-$(deg8_jobs) ./slurm_eval_degradation.sh $(deg8_tasks)

task_file="tasks-degradation-14.txt"
n_jobs = $(shell cat $(task_file) | wc -l )
deg_jobarray:
	@echo "running job array with $(n_jobs) jobs:"
	@cat $(task_file)
	sbatch --array=1-$(n_jobs) ./slurm_eval_degradation.sh $(task_file)


task_file="tasks-degradation-14.txt"
n_jobs = $(shell cat $(task_file) | wc -l )
sensitivityn_jobarray:
	@echo "running job array with $(n_jobs) jobs:"
	@cat $(task_file)
	sbatch --array=1-$(n_jobs) ./slurm_eval_sensitivity_n.sh $(task_file)


test_sensitivityn_tasks="test-tasks-sensitivityn-8.txt"
test_sensitivityn_jobs = $(shell cat $(test_sensitivityn_tasks) | wc -l )
test_sensitivityn_jobarray:
	@echo "running job array with $(test_sensitivityn_jobs) jobs:"
	@cat $(test_sensitivityn_tasks)
	sbatch --array=1-$(test_sensitivityn_jobs) ./slurm_eval_sensitivity_n.sh $(test_sensitivityn_tasks)

sensitivityn8_tasks="tasks-sensitivityn-8.txt"
sensitivityn8_jobs = $(shell cat $(sensitivityn8_tasks) | wc -l )
sensitivityn8_jobarray:
	@echo "running job array with $(sensitivityn_jobs) jobs:"
	@cat $(sensitivityn8_tasks)
	sbatch --array=1-$(sensitivityn8_jobs) ./slurm_eval_sensitivity_n.sh $(sensitivityn8_tasks)

sensitivityn1_tasks="tasks-sensitivityn-1.txt"
sensitivityn1_jobs = $(shell cat $(sensitivityn1_tasks) | wc -l )
sensitivityn1_jobarray:
	@echo "running job array with $(sensitivityn_jobs) jobs:"
	@cat $(sensitivityn8_tasks)
	sbatch --array=1-$(sensitivityn1_jobs) ./slurm_eval_sensitivity_n.sh $(sensitivityn1_tasks)

run_all_jobarray: sensitivityn8_jobarray sensitivityn1_jobarray deg14_jobarray deg8_jobarray bbox_jobarray



sensitivityn_redo="tasks-sensitivity-redo-occlusion-14x14.txt"
sensitivityn_redo_jobs = $(shell cat $(sensitivityn_redo) | wc -l )

sensitivityn_jobarray_redo_file:
	@echo "running job array with $(sensitivityn_redo_jobs) jobs:"
	@cat $(sensitivityn_redo)
	sbatch --array=1-$(sensitivityn_redo_jobs) ./slurm_eval_sensitivity_n.sh $(sensitivityn_redo)
