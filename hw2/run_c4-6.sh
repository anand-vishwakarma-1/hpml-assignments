# c4
# workers = 1
sbatch --job-name=lab2_c4_1 -o ./output/lab2_c4_1.out --partition=rtx8000 --gres=gpu:1 --cpus-per-task=20 --export=CUDA=false,WORKERS=1,OPT=sgd,EPOCHS=5,OUT=lab2_c4_1 train.sbatch


# c5
# workers = 12
sbatch --job-name=lab2_c5_gpu_sgd -o ./output/lab2_c5_gpu_sgd.out --gres=gpu:1 --partition=rtx8000 --cpus-per-task=20 --export=CUDA=true,WORKERS=4,OPT=sgd,EPOCHS=5,OUT=lab2_c5_gpu_sgd train.sbatch

# c6
# workers = 12 , opt = sgdn
sbatch --job-name=lab2_c6_sgdn -o ./output/lab2_c6_sgdn.out --gres=gpu:1 --partition=rtx8000 --cpus-per-task=20 --export=CUDA=true,WORKERS=4,OPT=sgdn,EPOCHS=5,OUT=lab2_c6_sgdn train.sbatch

# workers = 12 , opt = adagrad
sbatch --job-name=lab2_c6_adagrad -o ./output/lab2_c6_adagrad.out --gres=gpu:1 --partition=rtx8000 --cpus-per-task=20 --export=CUDA=true,WORKERS=4,OPT=adagrad,EPOCHS=5,OUT=lab2_c6_adagrad train.sbatch

# workers = 12 , opt = adadelta
sbatch --job-name=lab2_c6_adadelta -o ./output/lab2_c6_adadelta.out --gres=gpu:1 --partition=rtx8000 --cpus-per-task=20 --export=CUDA=true,WORKERS=4,OPT=adadelta,EPOCHS=5,OUT=lab2_c6_adadelta train.sbatch

# workers = 12 , opt = adam
sbatch --job-name=lab2_c6_adam -o ./output/lab2_c6_adam.out --gres=gpu:1 --partition=rtx8000 --cpus-per-task=20 --export=CUDA=true,WORKERS=4,OPT=adam,EPOCHS=5,OUT=lab2_c6_adam train.sbatchch