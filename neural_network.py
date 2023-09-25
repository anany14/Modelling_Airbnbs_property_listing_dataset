from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tabular_data import load_airbnb
import itertools
import json
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import List, Tuple, Dict, Any


class AirbnbNightlyPriceRegressionDataset(Dataset):
    """
    Dataset class for Airbnb nightly price regression.
    
    Args:
        Dataset
    """

    def __init__(self, df: pd.DataFrame = None) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
            df (pd.DataFrame, optional): The Airbnb dataset as a DataFrame. Defaults to None.
        """
        super().__init__()
        if df is None:
            df = pd.read_csv('airbnb-property-listing/tabular_data/clean_listing.csv', index_col='Unnamed: 0')
        #self.X, self.y = load_airbnb(df,label='Price_Night')
        self.X, self.y = self.preprocess_data(load_airbnb(df, label='Price_Night'))
        print(self.X.shape)

    def preprocess_data(self, data: Tuple[pd.DataFrame, pd.Series]) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = data
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        
        X_scaled = scaler.fit_transform(X) 

        return pd.DataFrame(X_scaled, columns=X.columns), y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return torch.tensor(self.X.iloc[index]), torch.tensor(self.y.iloc[index])


def split_data(dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split the dataset into training, validation, and test sets.

    Parameters
    ----------
        dataset (Dataset): The Airbnb dataset.

    Returns
    -------
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and test datasets.
    """
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.875), len(dataset) - int(len(dataset) * 0.875)])
    train_dataset, validation_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.85), len(train_dataset) - int(len(train_dataset) * 0.85)])

    print(f"\tTraining: {len(train_dataset)}")
    print(f"\tValidation: {len(validation_dataset)}")
    print(f"\tTesting: {len(test_dataset)}")

    return train_dataset, validation_dataset, test_dataset


class NN(torch.nn.Module):
    """
    Neural network class for regression.

    Args:
        torch.nn.Module
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the neural network.

        Parameters
        ----------
            config (dict): Configuration parameters for the network.
        """
        super().__init__()
        self.hidden_layer_width = config['hidden_layer_width']
        self.dropout = config['dropout']
        self.layers = torch.nn.Sequential(
            nn.Linear(11, self.hidden_layer_width),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer_width,1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters
        ----------
            X (torch.Tensor): Input data.

        Returns
        -------
            torch.Tensor: Output of the neural network.
        """
        return self.layers(X)


def get_nn_config(config_file: str = 'nn_config.yaml') -> dict:
    """
    Load neural network configuration from a YAML file.

    Parameters
    ----------
        config_file (str, optional): Path to the configuration file. Defaults to 'nn_config.yaml'.

    Returns
    -------
        dict: Loaded configuration.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_nn_config() -> List[Dict[str, Any]]:
    """
    Generate configurations for hyperparameter search.

    Parameters
    ----------
        None

    Returns
    -------
        List[Dict[str, Any]]: A list of dictionaries representing different hyperparameter configurations.
    """
    configs = []
    config = {
        'optimiser': ['SGD', 'Adam'],
        'learning_rate': [0.01,0.001],
        'hidden_layer_width': [12, 14, 16],
        'dropout':[0.2,0.3]
    }

    for hyperparam_values in itertools.product(*config.values()):
        hyperparam_dict = dict(zip(config.keys(), hyperparam_values))
        configs.append(hyperparam_dict)

    return configs


def train(train_loader: DataLoader, validation_loader: DataLoader, config: dict, epochs: int = 10) -> Tuple[NN, float, float, float, float, float]:
    """
    Train the neural network.

    Parameters
    ----------
        train_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        config (dict): Configuration parameters for training.
        epochs (int, optional): Number of training epochs. Defaults to 10.

    Returns
    -------
        Tuple[NN, float, float, float, float, float]: Trained model and training metrics.

    """
    model = NN(config)
    optimiser_name = config['optimiser']
    optimiser_class = getattr(torch.optim, optimiser_name)
    optimiser = optimiser_class(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter()
    batch_idx = 0
    train_rmse_loss = 0
    val_rmse_loss = 0
    training_r2 = 0
    validation_r2 = 0

    start_time = time.time()

    for epoch in range(epochs):
        for batch in train_loader:
            X_train, y_train = batch
            X_train = X_train.type(torch.float32)
            y_train = y_train.type(torch.float32)
            y_train = y_train.view(-1, 1)
            train_prediction = model(X_train)
            train_loss = F.mse_loss(train_prediction, y_train)
            train_loss = train_loss.type(torch.float32)
            train_loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('training_loss', train_loss.item(), batch_idx)
            batch_idx += 1
            rmse_loss = torch.sqrt(train_loss)
            train_rmse_loss += rmse_loss.item()
            train_r2 = r2_score(y_train.detach().numpy(), train_prediction.detach().numpy())
            #writer.add_scalar('training_r2', train_r2, batch_idx)
            training_r2 += train_r2

        for batch in validation_loader:
            X_val, y_val = batch
            X_val = X_val.type(torch.float32)
            y_val = y_val.type(torch.float32)
            y_val = y_val.view(-1, 1)
            val_prediction = model(X_val)
            val_loss = F.mse_loss(val_prediction, y_val)
            val_loss = val_loss.type(torch.float32)
            writer.add_scalar('validation_loss', val_loss.item(), batch_idx)
            rmse_loss = torch.sqrt(val_loss)
            val_rmse_loss += rmse_loss.item()
            val_r2 = r2_score(y_val.detach().numpy(), val_prediction.detach().numpy())
            #writer.add_scalar('validation_r2', val_r2, batch_idx)
            validation_r2 += val_r2

    end_time = time.time()
    training_duration = end_time - start_time

    train_rmse_loss = train_rmse_loss / (epochs * len(train_loader))
    val_rmse_loss = val_rmse_loss / (epochs * len(validation_loader))
    training_r2 = training_r2 / (epochs * len(train_loader))
    validation_r2 = validation_r2 / (epochs * len(validation_loader))

    return model, train_rmse_loss, val_rmse_loss, training_r2, validation_r2, training_duration


def test_model(model: NN, test_loader: DataLoader) -> Tuple[float, float, float]:
    """
    Test the trained neural network.

    Parameters
    ----------
        model (NN): Trained neural network.
        test_loader (DataLoader): DataLoader for test data.

    Returns
    -------
        Tuple[float, float, float]: Test RMSE loss, R-squared score, and inference latency.
    """
    writer = SummaryWriter()
    batch_idx = 0
    test_rmse_loss = 0
    testing_r2 = 0
    total_pred_time = 0

    for batch in test_loader:
        X_test, y_test = batch
        X_test = X_test.type(torch.float32)
        y_test = y_test.type(torch.float32)
        y_test = y_test.view(-1, 1)
        pred_start_time = time.time()
        test_prediction = model(X_test)
        pred_end_time = time.time()
        test_loss = F.mse_loss(test_prediction, y_test)
        test_loss = test_loss.type(torch.float32)
        writer.add_scalar("test_loss", test_loss, batch_idx)
        rmse_loss = torch.sqrt(test_loss)
        test_rmse_loss += rmse_loss.item()
        test_r2 = r2_score(y_test.detach().numpy(), test_prediction.detach().numpy())
        #writer.add_scalar("test_r2", test_r2, batch_idx)
        testing_r2 += test_r2
        total_pred_time += (pred_end_time - pred_start_time)
        batch_idx += 1

    test_rmse_loss = test_rmse_loss / len(test_loader)
    testing_r2 = testing_r2 / len(test_loader)
    inference_latency = total_pred_time / len(test_loader)

    return test_rmse_loss, testing_r2, inference_latency


def save_model(model: NN, hyperparameters: dict, performance_metrics: dict, file_path: str = "models/neural_networks/regression/price_night") -> None:
    """
    Save the trained neural network model.

    Parameters
    ----------
        model (NN): Trained neural network.
        hyperparameters (dict): Hyperparameters used for training.
        performance_metrics (dict): Performance metrics of the trained model.
        file_path (str, optional): File path to save the model. Defaults to "models/neural_networks/regression".

    Returns
    -------
        None

    """
    save_path = os.path.join(file_path, datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

    with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    with open(os.path.join(save_path, 'performance_metrics.json'), 'w') as f:
        json.dump(performance_metrics, f, indent=4)


def find_best_nn() -> None:
    """
    Find the best neural network configuration and train the model.
    """

    dataset = AirbnbNightlyPriceRegressionDataset()
    batch_size = 16
    train_dataset, validation_dataset, test_dataset = split_data(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    best_model = None
    best_hyperparameters = None
    best_rmse = float('inf')

    config = generate_nn_config()

    for index, element in enumerate(config):
        print(f"Training model {index + 1}/{len(config)} with config: {element}")
        model, train_rmse_loss, val_rmse_loss, training_r2, validation_r2, training_duration = train(train_loader, validation_loader, element)

        if val_rmse_loss < best_rmse:
            best_rmse = val_rmse_loss
            best_model = model
            best_hyperparameters = element
            test_rmse_loss, testing_r2, inference_latency = test_model(model, test_loader)
            performance_metrics = {
                "RMSE_loss": {
                    "training": train_rmse_loss,
                    "validation": val_rmse_loss,
                    "test": test_rmse_loss
                },
                "R_squared": {
                    "training": training_r2,
                    "validation": validation_r2,
                    "test": testing_r2
                },
                "training_duration": training_duration,
                "inference_latency": inference_latency
            }

    print(f"Best Model is {best_model},\t {performance_metrics}")
    save_model(best_model, best_hyperparameters, performance_metrics)


if __name__ == "__main__":
    find_best_nn()
