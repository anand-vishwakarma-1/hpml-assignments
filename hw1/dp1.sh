#!/bin/bash
#SBATCH --job-name=lab1_task1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --output=%x.out


gcc -O3 -Wall -o dp1 dp1.c
./dp1 1000000 1000
./dp1 300000000 20
# ls -l -a