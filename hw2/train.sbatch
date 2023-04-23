#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END
#SBATCH --mail-user=asv8775@nyu.edu

if ${CUDA}
then

echo $cuda

singularity exec --nv \
	    --overlay /scratch/asv8775/pytorch-example/pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 train.py --workers ${WORKERS} --opt ${OPT} --epochs ${EPOCHS} --cuda"

else


singularity exec \
	    --overlay /scratch/asv8775/pytorch-example/pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 train.py --workers ${WORKERS} --opt ${OPT} --epochs ${EPOCHS}"

fi