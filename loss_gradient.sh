python data_filter.py --filter_method loss_variance --sample_method slice

python data_filter.py --filter_method gradient_variance --sample_method slice

python main.py --dataset CIFAR10 --ipc 1 --subset True --filter_method loss_variance --sample_method slice --wandb_name loss_variance

python main.py --dataset CIFAR10 --ipc 1 --subset True --filter_method gradient_variance --sample_method slice --wandb_name gradient_variance

python main.py --dataset CIFAR10 --ipc 10 --subset True --filter_method loss_variance --sample_method slice  --wandb_name loss_variance

python main.py --dataset CIFAR10 --ipc 10 --subset True --filter_method gradient_variance --sample_method slice --wandb_name gradient_variance