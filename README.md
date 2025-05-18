# Federated Learning Benchmark for Medical Image Classification

This repository provides a comprehensive benchmark for federated learning (FL) in medical image classification tasks. It is designed to evaluate various FL algorithms on real medical image datasets. The benchmark corresponds to the paper "[Federated Learning for Medical Image Classification: A Comprehensive Benchmark](https://arxiv.org/abs/2504.05238)".

## Features

- **Traditional Federated Learning Algorithms**: Implemented in `fed_train_main.py`

- **Personalized Federated Learning Algorithms**: Implemented in `personalized_fed_train.py` and `distill_fed_train.py`

- **Elastic Aggregation**: Implemented in `elastic_aggregation.py`

- **DDPM-Augmented Federated Learning**: Our proposed method, which optimizes FL training using DDPM-augmented datasets, is implemented in `fed_train_with_gen.py`. The DDPM training and data generation code is located in the `DDPM` directory.

## Installation

To set up the environment, please install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage
<!-- ### Traditional Federated Learning -->

To run the traditional federated learning algorithms, use the following command:

```bash
python fed_train_main.py
 --dataset <DATASET_NAME> --class_num <NUM_CLASSES> --num_clients <NUM_CLIENTS> --local_epochs <LOCAL_EPOCHS> --global_epochs <GLOBAL_EPOCHS> --resolution <IMAGE_RESOLUTION> --algorithm <ALGORITHM> [OPTIONS]
```

#### Key Arguments:
- `--lr`: Learning rate for the optimizer.
- `--resume`, `-r`: Resume training from a checkpoint.
- `--data_save`: Use a pre-saved dataset file.
- `--dataset`: Name of the dataset to use.
- `--class_num`: Number of classes in the dataset.
- `--num_clients`: Number of clients in the FL system.
- `--local_epochs`: Number of local training epochs for each client.
- `--global_epochs`: Number of global training epochs.
- `--resolution`: Resolution of the input images.
- `--algorithm`: Federated learning algorithm to use (e.g., `avg`, `fedprox`, `fednova`, etc.).
- `--mu`: Parameter used by FedProx/MOON algorithms.
- `--rho`: Parameter used by FedNova algorithm.
- `--restrict`: Parameter used by FedRS algorithm.
- `--non_iid`: Use non-IID data distribution.
- `--use_f1`: Evaluate the model using F1 score.

#### Example Usage:
To run the script with the NeoJaundice dataset, 2 classes, 5 clients, 5 local epochs, 100 global epochs, 256x256 image resolution, and the FedProx algorithm, use:

```bash
python fed_train_main.py
 --dataset NeoJaundice --class_num 2 --num_clients 5 --local_epochs 5 --global_epochs 100 --resolution 256 --algorithm fedprox --mu 0.01
```


## Citation
If you find this repository helpful, please cite our paper:
```
@article{zhou2025federated,
  title={Federated Learning for Medical Image Classification: A Comprehensive Benchmark},
  author={Zhou, Zhekai and Luo, Guibo and Chen, Mingzhi and Weng, Zhenyu and Zhu, Yuesheng},
  journal={arXiv preprint arXiv:2504.05238},
  year={2025}
}
```