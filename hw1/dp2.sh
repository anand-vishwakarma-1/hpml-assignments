#!/bin/bash
#SBATCH --job-name=lab1_task1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --output=%x.out


gcc -O3 -Wall -o dp2 dp2.c
./dp2 1000000 1000
./dp2 300000000 20
# ls -l -a