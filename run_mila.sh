#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=None
#SBATCH --mail-type=FAIL
#SBATCH --job-name=hm1
#SBATCH --array=0-0
#SBATCH --no-requeue


python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false


