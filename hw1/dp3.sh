#!/bin/bash
#SBATCH --job-name=lab1_task1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --output=%x.out

module load intel/19.1.2
g++ -I /share/apps/intel/19.1.2/mkl/include/ -L /share/apps/intel/19.1.2/mkl/lib/intel64/ -o dp3 dp3.c -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
./dp3 1000000 1000
./dp3 300000000 20
# ls -l -a