import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tabular_data import load_airbnb
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import yaml
import time
from datetime import datetime
import os
import json

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self,df = pd.read_csv('airbnb-property-listing/tabular_data/clean_listing.csv',index_col='Unnamed: 0')) -> None:
        super().__init__()
        self.X,self.y = load_airbnb(df,label="Price_Night")
        print(self.X.shape)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return torch.tensor(self.X.iloc[index]), torch.tensor(self.y.iloc[index])
    

def split_data(dataset):
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset)* 0.875), len(dataset)-int(len(dataset)*0.875)])

    train_dataset,validation_dataset = random_split(train_dataset, [int(len(train_dataset)*0.85), len(train_dataset)-int(len(train_dataset)*0.85)])

    print(f"\tTraining: {len(train_dataset)}")
    print(f"\tValidation: {len(validation_dataset)}")
    print(f"\tTesting: {len(test_dataset)}")

    return train_dataset, validation_dataset, test_dataset


class NN(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        # Define layers and architecture here
        self.hidden_layer_width = config['hidden_layer_width']
        self.layers = torch.nn.Sequential(
        nn.Linear(11,self.hidden_layer_width),
        nn.ReLU(),
        nn.Linear(self.hidden_layer_width,1)
        )

    def forward(self, X):
        return self.layers(X)
    

def get_nn_config(config_file='nn_config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(train_loader,validation_loader,config,epochs=10):

    #scaler = MinMaxScaler()
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

    for epoch in range (epochs):
        for batch in train_loader:
            X_train,y_train = batch
            X_train = X_train.type(torch.float32)
            #X_train = torch.tensor(scaler.fit_transform(X_train))
            y_train = y_train.type(torch.float32)
            y_train = y_train.view(-1,1)
            train_prediction = model(X_train)
            train_loss = F.mse_loss(train_prediction,y_train)
            train_loss = train_loss.type(torch.float32)
            train_loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('training_loss',train_loss.item(),batch_idx)
            batch_idx += 1
            rmse_loss = torch.sqrt(train_loss)
            train_rmse_loss += rmse_loss.item()
            train_r2 = r2_score(y_train.detach().numpy(), train_prediction.detach().numpy())
            writer.add_scalar('training_r2', train_r2, batch_idx)
            training_r2 += train_r2
  

        for batch in validation_loader:
            X_val,y_val = batch
            X_val = X_val.type(torch.float32)
            #X_val = torch.tensor(scaler.fit_transform(X_val))
            y_val = y_val.type(torch.float32)
            y_val = y_val.view(-1, 1)
            val_prediction = model(X_val)
            val_loss = F.mse_loss(val_prediction,y_val)
            val_loss = val_loss.type(torch.float32)
            writer.add_scalar('validation_loss',val_loss.item(),batch_idx)
            rmse_loss = torch.sqrt(val_loss)
            val_rmse_loss += rmse_loss.item()

            val_r2 = r2_score(y_val.detach().numpy(), val_prediction.detach().numpy())
            writer.add_scalar('validation_r2', val_r2, batch_idx)
            validation_r2 += val_r2

    end_time = time.time()
    training_duration = end_time - start_time

    train_rmse_loss = train_rmse_loss/(epochs*len(train_loader))
    val_rmse_loss = val_rmse_loss/(epochs*len(validation_loader))
    training_r2 = training_r2/(epochs*len(train_loader))
    validation_r2 = validation_r2/(epochs*len(validation_loader))


    return model,train_rmse_loss,val_rmse_loss,training_r2,validation_r2,training_duration


def test_model(model,test_loader):
    
    writer = SummaryWriter()
    batch_idx = 0
    test_rmse_loss = 0
    testing_r2 = 0
    total_pred_time = 0

    for batch in test_loader:
        X_test,y_test = batch
        X_test = X_test.type(torch.float32)
        y_test = y_test.type(torch.float32)
        y_test = y_test.view(-1,1)
        pred_start_time = time.time()
        test_prediction = model(X_test)
        pred_end_time = time.time()
        test_loss = F.mse_loss(test_prediction,y_test)
        test_loss = test_loss.type(torch.float32)
        writer.add_scalar("test_loss",test_loss,batch_idx)
        rmse_loss = torch.sqrt(test_loss)
        test_rmse_loss += rmse_loss.item()

        test_r2 = r2_score(y_test.detach().numpy(),test_prediction.detach().numpy())
        writer.add_scalar("test_r2",test_r2,batch_idx)
        testing_r2 += test_r2
        
        total_pred_time += (pred_end_time - pred_start_time)
        batch_idx += 1 

    test_rmse_loss = test_rmse_loss/len(test_loader)
    testing_r2 = testing_r2/len(test_loader)
    inference_latency = total_pred_time/len(test_loader)

    return test_rmse_loss,testing_r2,inference_latency


def save_model(model,hyperparameters,performance_metrics,file_path="models\\neural_networks\\regression"):
    """Saves the trained neural network to a file path with name as hyperparameter values and performance metrics."""

    save_path = os.path.join(file_path,datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_path,'model.pt'))

    with open(os.path.join(save_path,'hyperparameters.json'),'w') as f:
        json.dump(hyperparameters,f,indent=4)

    with open(os.path.join(save_path,'performance_metrics.json'),'w') as f:
        json.dump(performance_metrics,f,indent=4)


def main():
    dataset = AirbnbNightlyPriceRegressionDataset()
    batch_size = 16
    train_dataset, validation_dataset, test_dataset = split_data(dataset)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

    config = get_nn_config()
    model,train_rmse_loss,val_rmse_loss,training_r2,validation_r2,training_duration = train(train_loader,validation_loader,config)
    print(f"{train_rmse_loss}\t{val_rmse_loss}\t{training_r2}\t{validation_r2}\t{training_duration}\n")
    test_rmse_loss,testing_r2,inference_latency = test_model(model,test_loader)
    print(f"{test_rmse_loss}\t{testing_r2}\t{inference_latency}")

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

    save_model(model,config,performance_metrics)
    


main()