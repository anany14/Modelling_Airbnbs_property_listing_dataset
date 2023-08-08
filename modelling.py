from tabular_data import load_airbnb
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import itertools
import joblib
import os
import json
from typing import List, Tuple
from typing import Any, Dict, List, Tuple


# loading the clean dataset as features and labels
df = pd.read_csv("C:/Users/Anany/OneDrive/Desktop/Github/AIcore/Modelling_Airbnbs_property_listing_dataset/airbnb-property-listing/tabular_data/clean_listing.csv")
X, y = load_airbnb(df, label="Price_Night")

# splitting dataset into training set and testing set 80:20 ratio and then 50:50 into testing and validation sets
# respectively for cross-validation purposes later on in model evaluation process
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=42)
# printing the shapes of features and labels
print("Training data shape:", X_train.shape)
print("Training Label shape:", y_train.shape)

# no need to create an instance of linear regression algorithm as its already done in the parameters
model = SGDRegressor(random_state=42)  
# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute RMSE for training and test sets
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
# Compute R^2 (coefficient of determination) for training and test sets
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print the results
print("Initial Training RMSE:", train_rmse)
print("Initial Test RMSE:", test_rmse)
print("Initial Training R-squared (R2):", train_r2)
print("Initial Test R-squared (R2):", test_r2)
#the outputs at this point suggest that there might be some issues with modelling
#extreme high values of mse and negative r-squared suggest that model is doing poorly and not fitting well


def custom_tune_regression_model_hyperparameters(
    model_class: type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameters: Dict[str, list]
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Perform a grid search over a range of hyperparameter values for a given regression model.

    Parameters:
        model_class (class): The class of the regression model.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        X_val (DataFrame): Validation features.
        y_val (Series): Validation labels.
        X_test (DataFrame): Test features.
        y_test (Series): Test labels.
        hyperparameters (dict): A dictionary of hyperparameter names mapping to a list of values to be tried.

    Returns:
        Tuple containing:
            - best_model: The best regression model.
            - best_hyperparameters (dict): The best hyperparameters found during the grid search.
            - performance_metrics (dict): A dictionary of performance metrics, including "validation_RMSE" and "test_RMSE".
    """
    best_model = None
    best_hyperparameters = {}
    best_val_rmse = float("inf")
    
    # Iterate through all combinations of hyperparameter values
    for hyperparam_values in itertools.product(*hyperparameters.values()):
        hyperparam_dict = dict(zip(hyperparameters.keys(), hyperparam_values))
        model = model_class(**hyperparam_dict,random_state=42)
        model.fit(X_train, y_train)
        
        y_val_pred = model.predict(X_val)
        val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        
        # Update best model if validation RMSE improves
        if val_rmse < best_val_rmse:
            best_model = model
            best_hyperparameters = hyperparam_dict
            best_val_rmse = val_rmse
            
    # Train the best model on the full training and validation sets
    best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    
    # Calculate final validation RMSE
    final_val_pred = best_model.predict(X_val)
    final_val_rmse = mean_squared_error(y_val, final_val_pred, squared=False)
    
    # Calculate final test RMSE
    final_test_pred = best_model.predict(X_test)
    final_test_rmse = mean_squared_error(y_test, final_test_pred, squared=False)
    
    # Store performance metrics
    performance_metrics = {
        "validation_RMSE": final_val_rmse,
        "test_RMSE": final_test_rmse
    }

    return best_model, best_hyperparameters, performance_metrics



def tune_regression_model_hyperparameters(model_class: type, X_train: pd.DataFrame, y_train: pd.Series, hyperparameters: List[Tuple[str, list]], cv_folds: int = 30) -> GridSearchCV:
    """
    Perform a grid search over a range of hyperparameters for a given regression model using GridSearchCV.

    Parameters:
        model_class (class): The class of the regression model.
        X_train (pd.DataFrame): Training features with shape [n_samples, n_features].
        y_train (pd.Series): Training labels with shape [n_samples].
        hyperparameters (list of tuples): List of tuples containing name-value pairs of hyperparameters to search.
        cv_folds (int): Number of cross-validation folds.


    Returns:
        GridSearchCV: A grid search object containing the best estimator and hyperparameters.
    """
    model = model_class(random_state=42)
    gridsearch = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=cv_folds, scoring='neg_root_mean_squared_error', verbose=1)
    gridsearch.fit(X_train, y_train)
    return gridsearch



def save_model(model,hyperparametrics,performance_metrics,folder='models/regression/linear_regression'):

    """
    Saves the trained regression model, its hyperparameters and performance metrics in a specified directory.

    Parameters:
        model: The trained regression model.
        hyperparametrics (dict): dictionary containing all of the parameters used for training.
        performance_metrics (dict): dictionary containing all the performance metrics.
        folder (str): The folders where the file will be saved.
    """

    # create folder if it doesn't exist
    os.makedirs(folder,exist_ok=True)

    # save model using joblib
    model_name = os.path.join(folder,'model.joblib')
    joblib.dump(model, model_name)

    # save hyperparameters to a json file
    hyperparametrics_name = os.path.join(folder,'hyperparametrics.json')
    with open(hyperparametrics_name, 'w') as f:
        json.dump(hyperparametrics,f,indent=4)

    # save performance_metrics to a json file
    performance_metrics_name = os.path.join(folder,'performance_metrics.json')
    with open(performance_metrics_name, 'w') as f:
        json.dump(performance_metrics,f,indent=4)

