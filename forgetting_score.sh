python main.py --dataset CIFAR10 --ipc 1 --subset True --filter_method forgetting --sample_method slice --wandb_name forgetting

python main.py --dataset CIFAR10 --ipc 10 --subset True --filter_method forgetting --sample_method slice --wandb_name forgetting

python main.py --dataset CIFAR10 --ipc 10 --subset False --filter_method forgetting --sample_method slice --wandb_name baseline