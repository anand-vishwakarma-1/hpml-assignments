#!/bin/bash
#SBATCH --job-name=lab1_task1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=4GB
#SBATCH --output=%x.out


singularity exec --bind /home/asv8775/hw1:/home/hw1 \
	    --overlay /scratch/asv8775/pytorch-example/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python3 /home/hw1/dp4.py 1000000 1000; python3 /home/hw1/dp4.py 300000000 20; python3 /home/hw1/dp5.py 1000000 1000; python3 /home/hw1/dp5.py 300000000 20;ls /home/hw1/"
# ls -l -a