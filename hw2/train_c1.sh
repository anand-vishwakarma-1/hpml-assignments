#!/bin/bash
singularity exec \
	    --overlay /scratch/asv8775/pytorch-example/pytorch.ext3:ro \
	    //scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 temp.py > temp.txt"