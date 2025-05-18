import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from trainer.datasets.data_preparation import DataPreparation
from trainer.datasets.classification_dataset import ClassificationDataset
from trainer.model import create_model
from trainer.mlp_model import MLP

import os
import json

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, batch_size, train_loader, test_loader, epochs, device):
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

def train_supervised_model(config: dict):
    # --- read parameters ---
    csv_path = config['csv_path']
    target_columns = config['target_column'] if isinstance(config['target_column'], list) else [config['target_column']]
    model_class = config['model_class']
    archi_info = config['model_arch']
    learning_rate = config.get('learning_rate', 0.001)
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 32)
    test_fraction = config.get('test_fraction', 0.2)
    cleaning = config.get('cleaning', False)
    seed = config.get('seed', 42)
    device = config.get('device', 'cpu')
    image_column = config.get('image_column', None)

    # --- data prep ---
    data_prep = DataPreparation(
        csv_path,
        fraction=test_fraction,
        cleaning=cleaning,
        seed=seed,
        image_column=image_column
    )
    data_prep.read_data()
    data_prep.extract_cols(target_columns)
    data_prep.split()
    train_set, test_set = data_prep.get_train_test()
    classes = data_prep.get_classes()

    train_ds = ClassificationDataset(train_set, classes, image_column=image_column)
    test_ds = ClassificationDataset(test_set, classes, image_column=image_column)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # --- model handling---
    model = create_model(model_class, archi_info, num_classes=len(classes))
    model.to(device)

    # --- cost/opti ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- "real" training ---
    trainer = Trainer(model, criterion, optimizer)
    trainer.train(batch_size, train_loader, test_loader, epochs, device)

    # --- saving ---
    save_dir = config.get('save_dir', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)

    archi_path = os.path.join(save_dir, 'model_architecture.json')
    with open(archi_path, 'w') as f:
        json.dump(archi_info, f, indent=4)

    print(f"\nTraining complete. Model saved to: {model_path}\nArchitecture saved to: {archi_path}")