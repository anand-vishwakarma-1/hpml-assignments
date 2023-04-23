#!/bin/bash
#SBATCH --job-name=lab2_c1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --account=ece_gy_9143-2023sp
#SBATCH --partition=n2c48m24
#SBATCH --output=lab2_c1.txt

singularity exec --nv --bind /home/asv8775/hw2:/home/hw2 \
	    --overlay /scratch/asv8775/pytorch-example/pytorch.ext3:ro \
	    /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 c1.py | tee /home/hw2/lab2_c1.txt"