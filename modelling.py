from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from tabular_data import load_airbnb
from typing import List, Tuple, Dict, Type, Any
import itertools
import joblib
import json
import os
import pandas as pd

# Split dataset into training, testing, and validation sets
def split_X_y(X:pd.DataFrame, y:pd.Series)->Tuple:
    """
    Split the dataset into training,testing and validation tests.
    
    Parameters
    ----------

        X : (pd.DataFrame) \n\t Features.
        y : (pd.Series) \n\t Labels.

    Returns
    -------

        Tuple (X_train,y_train,X_test,y_test,X_val,y_val)    
        \nA tuple containing the split features and labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training Label shape: {y_train.shape}")
    print("Number of samples in:")
    print(f"    Training:   {len(y_train)}")
    print(f"    Testing:    {len(y_test)}")
    print(f"    Validation:   {len(y_val)}")
    
    return X_train, y_train, X_test, y_test, X_val, y_val

# Train a logistic regression model and return performance metrics
def train_regression_model(X_train:pd.DataFrame, y_train:pd.Series, X_test:pd.DataFrame, y_test:pd.Series, modelclass:type = SGDRegressor):
    """
    Train a regression model and print its initial performance on training and test sets.
 
    Parameters
    ----------

        X_train : (pd.DataFrame) \nFeatures for training set.
        Y_train : (pd.Series) \nLabels for training set.
        X_test : (pd.DataFrame) \nFeatures for testing set.
        Y_test : (pd.Series) \nLabel for testing set.
        modelclass : (type), default = SGDRegressor) \nThe class for the regression model.

    Return
    ------

        None
    """
    #Normalise features using MinMaxScaler if class is sgdregressor
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Create an instance of the regression algorithm
    model = modelclass(random_state=42)
    
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
    print(f"Initial Training RMSE: {train_rmse} | Initial Training R-squared (R2): {train_r2}")
    print(f"Initial Test RMSE: {test_rmse} | Initial Test R-squared (R2): {test_r2}")
    # The outputs at this point suggest that there might be some issues with the model
    # Extreme high values of MSE and negative R-squared suggest that the model is not fitting well


# Custom tune hyperparameters of a model and return the best model and metrics
def custom_tune_regression_model_hyperparameters(model_class: type,X_train: pd.DataFrame,y_train: pd.Series,X_val: pd.DataFrame,y_val: pd.Series,hyperparameters: Dict[str, List]):
    """
    Perform a grid search over a range of hyperparameter values for a given regression model.

    Parameters
    ----------

        model_class : (type) \nThe class of the regression model.
        X_train : (pd.DataFrame) \nTraining features.
        y_train : (pd.Series) \nTraining labels.
        X_val : (pd.DataFrame) \nValidation features.
        y_val : (pd.Series) \nValidation labels.
        hyperparameters : (dict[str,float]) \nA dictionary of hyperparameter names mapping to a list of values to be tried.

    Returns
    -------

        Tuple[
            \n best_model: The best regression model.
            \n best_hyperparameters (dict): The best hyperparameters found during the grid search.
            \n performance_metrics (dict): A dictionary of performance metrics, including "validation_RMSE" and "test_RMSE".]
    """
    best_model = None
    best_hyperparameters = {}
    best_val_rmse = float("inf")

    # Iterate through all combinations of hyperparameter values
    for hyperparam_values in itertools.product(*hyperparameters.values()):
        hyperparam_dict = dict(zip(hyperparameters.keys(), hyperparam_values))
        model = model_class(**hyperparam_dict, random_state=42)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

        # Update best model if validation RMSE improves
        if val_rmse < best_val_rmse:
            best_model = model
            best_hyperparameters = hyperparam_dict
            best_val_rmse = val_rmse

    # Train the best model on the full training and validation sets
    # best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    # Train the best model on the training set 
    best_model.fit(X_train,y_train)

    # Calculate final training rmse
    final_train_pred = best_model.predict(X_train)
    final_train_rmse = mean_squared_error(y_train, final_train_pred, squared=False)
    final_train_r2 = r2_score(y_train,final_train_pred)

    # Calculate final validation rmse
    final_val_pred = best_model.predict(X_val)
    final_val_rmse = mean_squared_error(y_val, final_val_pred, squared=False)
    final_val_r2 = r2_score(y_val,final_val_pred)

    # Store performance metrics
    performance_metrics = {"training_rmse": final_train_rmse,"training_r2":final_train_r2,"validation_rmse": final_val_rmse,"validation_r2":final_val_r2}
    return best_model, best_hyperparameters, performance_metrics

# Tune hyperparameters of a model and return the best model and metrics
def tune_regression_model_hyperparameters(model_class: type,X_train: pd.DataFrame,y_train: pd.Series,X_val : pd.DataFrame,y_val: pd.Series,
    hyperparameters: List[Tuple[str, List]],cv_folds: int = 5) -> GridSearchCV:
    """
    Perform a grid search over a range of hyperparameters for a given regression model using GridSearchCV.

    Parameters
    ----------

        model_class : (type) \nThe class of the regression model.
        X_train : (pd.DataFrame) \nTraining features with shape [n_samples, n_features].
        y_train : (pd.Series) \nTraining labels with shape [n_samples].
        X_val : (pd.DataFrame) \nValidation features.
        y_val : (pd.Series) \nValidation labels.
        hyperparameters : (list of tuples) \nList of tuples containing name-value pairs of hyperparameters to search.
        cv_folds : (int),default=5 \nNumber of cross-validation folds.

    Returns
    -------

        Tuple[Any, Dict[str, Any], Dict[str, float]] \n Tuple containing trained model,best parameters and performance metrics
    """
    if model_class == SGDRegressor:
        #Normalise features using MinMaxScaler if class is sgdregressor
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.fit_transform(X_val)

    performance_metrics = {}
    model = model_class(random_state=42)
    gridsearch = GridSearchCV(estimator=model,param_grid=dict(hyperparameters),cv=cv_folds,scoring='neg_root_mean_squared_error',verbose=1)

    gridsearch.fit(X_train, y_train)

    best_model = gridsearch.best_estimator_
    best_hyperparameters = gridsearch.best_params_

    y_val_pred = gridsearch.best_estimator_.predict(X_val)
    val_rmse = mean_squared_error(y_val,y_val_pred,squared=False)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"best hyperparameters: {best_hyperparameters}")
    print(f"gridsearch_rmse: {- gridsearch.best_score_}")
    print(f"validation_rmse: {val_rmse}")
    print(f"validation_r2: {val_r2}")

    performance_metrics['validation_rmse'] = val_rmse
    performance_metrics['gridsearch_rmse'] = -gridsearch.best_score_
    performance_metrics['validation_r2'] = val_r2

    return best_model, best_hyperparameters, performance_metrics


def save_model(model, hyperparameters, performance_metrics, folder='models/regression/linear_regression'):
    """
    Saves the trained regression model, its hyperparameters, and performance metrics in a specified directory.

    Parameters
    ----------

        model: (type) \nThe trained regression model.
        hyperparameters : (dict[str,float]) \nDictionary containing all of the parameters used for training.
        performance_metrics : (dict[str,float]) \nDictionary containing all the performance metrics.
        folder : (str),default='models/regression/linear_regression' \nThe folder where the files will be saved.

    Returns
    -------
        
        None
    """
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save model using joblib
    model_name = os.path.join(folder, 'model.joblib')
    joblib.dump(model, model_name)

    # Save hyperparameters to a json file
    hyperparameters_name = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_name, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Save performance_metrics to a json file
    performance_metrics_name = os.path.join(folder, 'performance_metrics.json')
    with open(performance_metrics_name, 'w') as f:
        json.dump(performance_metrics, f, indent=4)


def evaluate_all_models(X_train, y_train, X_val, y_val):
    """
    Evaluate different regression models using hyperparameter tuning and save results.

    Parameters
    ----------

        X_train : (pd.DataFrame) \nTraining features.
        y_train : (pd.Series) \nTraining labels.
        X_val : (pd.DataFrame) \nValidation features.
        y_val : (pd.Series) \nValidation labels.
    
    Hyperparameters for regression models
    ---------------

    \t``SGDRegressor`` - 
    \n``alpha``: This controls the regularization strength. Start with a small value like 0.0001 and gradually increase if needed.
    \n``max_iter``: The maximum number of iterations. A larger value, such as 1000, should be a reasonable starting point.
    \n``tol``: The tolerance for stopping criteria. You can start with 1e-3.
    
    \t``DecisionTreeRegressor`` - 
    \n``max_depth``: Maximum depth of the tree. Start with None (no depth limit) and then try smaller values like 5, 3, or 1.
    \n``min_samples_split``: Minimum number of samples required to split an internal node. Start with 2.
    \n``min_samples_leaf``: Minimum number of samples required to be at a leaf node. Start with 1.

    \t``RandomForestRegressor`` - 
    \n``n_estimators``: The number of trees in the forest. Start with a moderate number, such as 100, and tune from there.
    \n``max_depth``: Maximum depth of the individual trees. Similar to the decision tree, start with None.
    \n``min_samples_split``: Minimum number of samples required to split an internal node. Start with 2.
    \n``min_samples_leaf``: Minimum number of samples required to be at a leaf node. Start with 1.

    \t``GradientBoostingRegressor`` -
    \n``n_estimators``: The number of boosting stages (trees). Start with 100 or 150.
    \n``max_depth``: Maximum depth of the individual trees. Start with a smaller value like 3 or 5.
    \n``learning_rate``: The step size shrinkage used to prevent overfitting. Start with 0.1
    
    Returns
    -------

        None
    """

    # list of models to evaluate
    models_to_evaluate = [
    ("LinearRegression", SGDRegressor, {
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "max_iter": [1000, 2000, 5000],
            "tol": [1e-2, 1e-3, 1e-4]
    }),
    ("DecisionTree", DecisionTreeRegressor, {
            "max_depth": [5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [5, 10, 15]
    }),
    ("RandomForest", RandomForestRegressor, {
            "n_estimators": [300, 400, 500],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [5, 10, 15]
    }),
    ("GradientBoosting", GradientBoostingRegressor, {
            "learning_rate": [0.001, 0.01, 0.1],
            "max_depth": [3, 4, 5],
            "n_estimators": [300, 400, 500]
    })]

    for model_name, model_class, hyperparameters in models_to_evaluate:
        print(f"Evaluating {model_name}...")
        best_model, best_hyperparameters, performance_metrics = tune_regression_model_hyperparameters(
            model_class, X_train, y_train, X_val, y_val, hyperparameters)
        
        # Save model, hyperparameters, and metrics
        model_folder = f"models/regression/{model_name.lower()}"
        save_model(best_model, best_hyperparameters, performance_metrics, folder=model_folder)
        print(f"{model_name} evaluation complete.")


def find_best_model():
    """
    Find the best model based on the saved RMSE values from previously tuned models.

    Parameters
    ----------

        None

    Returns
    -------

        Tuple: best_model (model), best_hyperparameters (dict), best_rmse (float)
    """
    best_model = None
    best_hyperparameters = {}
    best_rmse = float("inf")

    # List of models to evaluate
    models_to_evaluate = ["LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting"]

    for model_name in models_to_evaluate:
        print(f"Evaluating {model_name}...")
        model_folder = f"models/regression/{model_name.lower()}"
        performance_metrics_path = os.path.join(model_folder, 'performance_metrics.json')

        with open(performance_metrics_path, 'r') as f:
            performance_metrics = json.load(f)

        model_rmse = performance_metrics.get("validation_rmse", float("inf"))

        if model_rmse < best_rmse:
            best_model = joblib.load(os.path.join(model_folder, 'model.joblib'))
            hyperparameters_path = os.path.join(model_folder, 'hyperparameters.json')
            with open(hyperparameters_path, 'r') as hp_file:
                best_hyperparameters = json.load(hp_file)
            best_rmse = model_rmse
            model_name_string = f"{model_name}"

    print("\n",f"{model_name_string} is the best model")
    return best_model, best_hyperparameters, performance_metrics

def main():
    """
    Runs all preprocessing steps and saves a trained model.
    
    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    # Load the clean dataset as features and labels
    df = pd.read_csv("C:/Users/Anany/OneDrive/Desktop/Github/AIcore/Modelling_Airbnbs_property_listing_dataset/airbnb-property-listing/tabular_data/clean_listing.csv")
    X, y = load_airbnb(df, label="Price_Night")

    # Split the data into training, testing, and validation sets
    X_train, y_train, X_test, y_test, X_val, y_val = split_X_y(X, y)

    # Train an initial regression model and print its performance
    train_regression_model(X_train, y_train, X_test, y_test)

    # Evaluate different regression models and save the best one
    evaluate_all_models(X_train, y_train, X_val, y_val)

    # Find the best model from the saved models
    best_model, best_hyperparameters, performance_metrics = find_best_model()
    print("\n Best Model:", best_model)
    print("\n Best Hyperparameters:", best_hyperparameters)
    print("\n Performance Metrics:", performance_metrics)

    """
    print("using custome tune function, the best paramteters for the best model is:")
    print("\n Best Model:", best_model)
    print("\n Best Hyperparameters:", best_hyperparameters)
    print("\n Performance Metrics:", performance_metrics)
    y_test_pred = best_model.predict(X_test)
    y_test_rmse = mean_squared_error(y_test,y_test_pred,squared=False)
    print("TEST RMSE:",y_test_rmse)
    print("PERFORMANCE (Test RMSE):",performance_metrics['test_rmse'])
    """

if __name__ == "__main__":
    main()