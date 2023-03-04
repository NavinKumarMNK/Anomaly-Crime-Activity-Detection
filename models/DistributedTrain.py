import os
import sys
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
import utils.utils as utils

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from models.EfficientNetb3 import AutoEncoder
from models.EfficientNetb3 import AutoEncoderDataset

from ray.util.sgd import DistributedDataParallel
from ray.train.torch import DistributedDataParallelPlugin, TorchTrainer
# Initialize a PyTorch Lightning trainer

# Define your training function
world_size = 2

def train(config, checkpoint_dir=None, data_dir=None):
    # Load your dataset
    dataset_params = utils.config_parse('AUTOENCODER_DATASET')
    train_data = AutoEncoderDataset(**dataset_params)
    # Initialize your PyTorch Lightning module
    module = AutoEncoder()
    # Initialize a PyTorch Lightning trainer
    trainer = Trainer(
        gpus=-1,
        accelerator='ddp',
        plugins=[DistributedDataParallelPlugin()],
        num_nodes=world_size,
        distributed_backend='ddp'
    )

    # Train the module using the PyTorch Lightning trainer
    trainer.fit(module, train_data)


# Define your configuration dictionary
config = {
    'lr': 0.001,
    'epochs': 10,
    'device': 0
}

# Initialize a TorchTrainer with the training function and configuration dictionary
trainer = TorchTrainer(
    training_function=train,
    config=config,
    backend="ray"
)

# Start the training process
trainer.train()