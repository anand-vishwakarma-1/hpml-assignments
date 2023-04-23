# c1 and c2
sbatch --job-name=lab2_c1-c2 -o ./output/lab2_c1-c2.out --partition=rtx8000 --gres=gpu:1 --cpus-per-task=20 --export=CUDA=false,WORKERS=0,OPT=sgd,EPOCHS=5,OUT=lab2_c1-c2 train.sbatch

# c3
# workers = 0
sbatch --job-name=lab2_c3_0 -o ./output/lab2_c3_0.out --partition=rtx8000 --gres=gpu:1 --cpus-per-task=20 --export=CUDA=false,WORKERS=0,OPT=sgd,EPOCHS=5,OUT=lab2_c3_0 train.sbatch

# workers = 4
sbatch --job-name=lab2_c3_4 -o ./output/lab2_c3_4.out --partition=rtx8000 --gres=gpu:1 --cpus-per-task=20 --export=CUDA=false,WORKERS=4,OPT=sgd,EPOCHS=5,OUT=lab2_c3_4 train.sbatch

# workers = 8
sbatch --job-name=lab2_c3_8 -o ./output/lab2_c3_8.out --partition=rtx8000 --gres=gpu:1 --cpus-per-task=20 --export=CUDA=false,WORKERS=8,OPT=sgd,EPOCHS=5,OUT=lab2_c3_8 train.sbatch

# workers = 12
sbatch --job-name=lab2_c3_12 -o ./output/lab2_c3_12.out --partition=rtx8000 --gres=gpu:1 --cpus-per-task=20 --export=CUDA=false,WORKERS=12,OPT=sgd,EPOCHS=5,OUT=lab2_c3_12 train.sbatch

# workers = 16
sbatch --job-name=lab2_c3_16 -o ./output/lab2_c3_16.out --partition=rtx8000 --gres=gpu:1 --cpus-per-task=20 --export=CUDA=false,WORKERS=16,OPT=sgd,EPOCHS=5,OUT=lab2_c3_16 train.sbatch



# sbatch --job-name=lab2_c3_20 -o ./output/lab2_c3_20.out --partition=rtx8000 --gres=gpu:1 --cpus-per-task=20 --export=CUDA=false,WORKERS=20,OPT=sgd,EPOCHS=5,OUT=lab2_c3_20 train.sbatch
