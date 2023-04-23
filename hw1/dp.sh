#!/bin/bash
#SBATCH --job-name=lab1_task1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=4GB
#SBATCH --output=%x.out


gcc -O3 -Wall -o dp1 dp1.c
./dp1 1000000 1000 > dpc.txt
./dp1 300000000 20 >> dpc.txt


gcc -O3 -Wall -o dp2 dp2.c
./dp2 1000000 1000 >> dpc.txt
./dp2 300000000 20 >> dpc.txt


module load intel/19.1.2
g++ -I /share/apps/intel/19.1.2/mkl/include/ -L /share/apps/intel/19.1.2/mkl/lib/intel64/ -o dp3 dp3.c -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
./dp3 1000000 1000 >> dpc.txt
./dp3 300000000 20 >> dpc.txt


# singularity exec --bind /home/asv8775/hw1:/home/hw1 \
# 	    --overlay /scratch/asv8775/pytorch-example/overlay-15GB-500K.ext3:ro \
# 	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
# 	    /bin/bash -c "source /ext3/env.sh; python3 /home/hw1/dp4.py 1000000 1000; python3 /home/hw1/dp4.py 300000000 20; python3 /home/hw1/dp5.py 1000000 1000; python3 /home/hw1/dp5.py 300000000 20;ls /home/hw1/"