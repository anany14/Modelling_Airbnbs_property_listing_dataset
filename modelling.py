from tabular_data import load_airbnb
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import itertools
import joblib
import os
import json
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler



def split_X_y(X, y):
    """
    Split the dataset into training, testing, and validation sets.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.

    Returns:
        Tuple: (X_train, y_train, X_test, y_test, X_val, y_val)
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


def train_regression_model(X_train, y_train, X_test, y_test, modelclass=SGDRegressor):
    """
    Train a regression model and print its initial performance on training and test sets.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        modelclass (type): The class ofthe regression model.

    Returns:
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
    # Extreme high values of MSE and negative R-squared suggest that the model is doing poorly and not fitting well


def custom_tune_regression_model_hyperparameters(
    model_class: type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameters: Dict[str, List]
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Perform a grid search over a range of hyperparameter values for a given regression model.

    Args:
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


def tune_regression_model_hyperparameters(
    model_class: type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val : pd.DataFrame,
    y_val   : pd.Series,
    hyperparameters: List[Tuple[str, List]],
    cv_folds: int = 5
) -> GridSearchCV:
    """
    Perform a grid search over a range of hyperparameters for a given regression model using GridSearchCV.

    Args:
        model_class (class): The class of the regression model.
        X_train (pd.DataFrame): Training features with shape [n_samples, n_features].
        y_train (pd.Series): Training labels with shape [n_samples].
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        hyperparameters (list of tuples): List of tuples containing name-value pairs of hyperparameters to search.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        GridSearchCV: A grid search object containing the best estimator and hyperparameters.
    """
    if model_class == SGDRegressor:
        #Normalise features using MinMaxScaler if class is sgdregressor
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.fit_transform(X_val)

    performance_metrics = {}
    model = model_class(random_state=42)
    gridsearch = GridSearchCV(
        estimator=model,
        param_grid=dict(hyperparameters),
        cv=cv_folds,
        scoring='neg_root_mean_squared_error',
        verbose=1
    )
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

    Args:
        model: The trained regression model.
        hyperparameters (dict): Dictionary containing all of the parameters used for training.
        performance_metrics (dict): Dictionary containing all the performance metrics.
        folder (str): The folder where the files will be saved.
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

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.

    Returns:
        None
    """
    # List of models to evaluate
    models_to_evaluate = [
    ("LinearRegression", SGDRegressor, {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "max_iter": [500, 1000, 2000, 3000, 4000],
        "tol": [1e-2, 1e-3, 1e-4, 1e-5]
    }),
    ("DecisionTree", DecisionTreeRegressor, {
        "max_depth": [None, 5, 3, 1],
        "min_samples_split": [2, 5, 7, 1],
        "min_samples_leaf": [10, 8, 6]
    }),
    ("RandomForest", RandomForestRegressor, {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 7, 10],
        "min_samples_leaf": [1, 2, 4, 6]
    }),
    ("GradientBoosting", GradientBoostingRegressor, {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1]
    })
    ]

    #just kept the earlier version
    models_to_evaluate = [
    ("LinearRegression", SGDRegressor, {
        "alpha": [0.001, 0.0001, 0.00001],  # Adding more alpha values for regularization
        "max_iter": [500, 1000, 2000],      # Trying higher values for max_iter
        "tol": [1e-2, 1e-3, 1e-4]            # No change, already a reasonable range
    }),
    ("DecisionTree", DecisionTreeRegressor, {
        "max_depth": [None, 5, 10, 15],      # Trying a higher value for max_depth
        "min_samples_split": [2, 5, 10],      # Increasing min_samples_split for more regularization
        "min_samples_leaf": [10, 8, 6]        # No change, already a reasonable range
    }),
    ("RandomForest", RandomForestRegressor, {
        "n_estimators": [300, 400, 500],      # Trying higher values for n_estimators
        "max_depth": [None, 10, 20],           # Limiting max_depth to control tree complexity
        "min_samples_split": [2, 5, 10],       # No change, already a reasonable range
        "min_samples_leaf": [6, 8, 10]         # No change, already a reasonable range
    }),
    ("GradientBoosting", GradientBoostingRegressor, {
        "n_estimators": [200, 300, 400],       # Trying higher values for n_estimators
        "max_depth": [3, 5, 7],                 # No change, already a reasonable range
        "learning_rate": [0.01, 0.005, 0.001]   # Trying lower values for learning_rate
    })
    ]

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

    """
    models_to_evaluate = [
    ("LinearRegression", SGDRegressor, {
        "alpha": [0.0001, 0.00001],      # Increasing alpha for more regularization
        "max_iter": [1000, 2000],         # Trying higher values for max_iter
        "tol": [1e-3, 1e-4]               # Keep the same range for tolerance
    }),
    ("DecisionTree", DecisionTreeRegressor, {
        "max_depth": [5, 10, 15, 20],      # Experiment with higher max_depth
        "min_samples_split": [2, 5, 10],    # Try different values for min_samples_split
        "min_samples_leaf": [10, 8, 6]      # Keep the same range for min_samples_leaf
    }),
    ("RandomForest", RandomForestRegressor, {
        "n_estimators": [300, 400, 500],    # Trying higher values for n_estimators
        "max_depth": [None, 10, 20],         # Limiting max_depth to control tree complexity
        "min_samples_split": [2, 5, 10],     # Keep the same range for min_samples_split
        "min_samples_leaf": [6, 8, 10]       # Keep the same range for min_samples_leaf
    }),
    ("GradientBoosting", GradientBoostingRegressor, {
        "n_estimators": [300, 400, 500],     # Trying higher values for n_estimators
        "max_depth": [3, 4, 5],               # Experiment with slightly higher max_depth
        "learning_rate": [0.05, 0.1]          # Trying a slightly higher learning rate
    })
    ]

    """

    for model_name, model_class, hyperparameters in models_to_evaluate:
        print(f"Evaluating {model_name}...")
        best_model, best_hyperparameters, performance_metrics = tune_regression_model_hyperparameters(
            model_class, X_train, y_train, X_val, y_val, hyperparameters
        )
        # Save model, hyperparameters, and metrics
        model_folder = f"models/regression/{model_name.lower()}"
        save_model(best_model, best_hyperparameters, performance_metrics, folder=model_folder)
        print(f"{model_name} evaluation complete.")


def find_best_model():
    """
    Find the best model based on the saved RMSE values from previously tuned models.

    Returns:
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




if __name__ == "__main__":
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




#what are hyper parameters

"""
Linear Regression (SGDRegressor):

alpha: This controls the regularization strength. Start with a small value like 0.0001 and gradually increase if needed.
max_iter: The maximum number of iterations. A larger value, such as 1000, should be a reasonable starting point.
tol: The tolerance for stopping criteria. You can start with 1e-3.
Decision Tree Regressor:

max_depth: Maximum depth of the tree. Start with None (no depth limit) and then try smaller values like 5, 3, or 1.
min_samples_split: Minimum number of samples required to split an internal node. Start with 2.
min_samples_leaf: Minimum number of samples required to be at a leaf node. Start with 1.
Random Forest Regressor:

n_estimators: The number of trees in the forest. Start with a moderate number, such as 100, and tune from there.
max_depth: Maximum depth of the individual trees. Similar to the decision tree, start with None.
min_samples_split: Minimum number of samples required to split an internal node. Start with 2.
min_samples_leaf: Minimum number of samples required to be at a leaf node. Start with 1.
Gradient Boosting Regressor:

n_estimators: The number of boosting stages (trees). Start with 100 or 150.
max_depth: Maximum depth of the individual trees. Start with a smaller value like 3 or 5.
learning_rate: The step size shrinkage used to prevent overfitting. Start with 0.1

"""


# feedback 2

"""
Training data shape: (712, 12)
Training Label shape: (712,)
Number of samples in:
    Training:   712
    Testing:    89
    Validation:   89
Initial Training RMSE: 98.11733497462616 | Initial Training R-squared (R2): 0.36777012858955604
Initial Test RMSE: 131.9735017931849 | Initial Test R-squared (R2): 0.32320274130570814
Evaluating LinearRegression...
Fitting 5 folds for each of 27 candidates, totalling 135 fits
best hyperparameters: {'alpha': 1e-05, 'max_iter': 500, 'tol': 0.01}
gridsearch_rmse: 97.7362631514105
validation_rmse: 127.4768534817222
validation_r2: -0.23837397130168658
LinearRegression evaluation complete.
Evaluating DecisionTree...
Fitting 5 folds for each of 36 candidates, totalling 180 fits
best hyperparameters: {'max_depth': 5, 'min_samples_leaf': 10, 'min_samples_split': 2}
gridsearch_rmse: 99.83932831904127
validation_rmse: 104.07966226324588
validation_r2: 0.174492747038261
DecisionTree evaluation complete.
Evaluating RandomForest...
Fitting 5 folds for each of 81 candidates, totalling 405 fits
best hyperparameters: {'max_depth': None, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 300}
gridsearch_rmse: 93.53765214999888
validation_rmse: 101.63984683370312
validation_r2: 0.212741880233853
RandomForest evaluation complete.
Evaluating GradientBoosting...
Fitting 5 folds for each of 27 candidates, totalling 135 fits
best hyperparameters: {'learning_rate': 0.005, 'max_depth': 3, 'n_estimators': 400}
gridsearch_rmse: 99.17506568871424
validation_rmse: 102.4967613110123
validation_r2: 0.1994113470784875
GradientBoosting evaluation complete.

Linear Regression:
The best hyperparameters found: {'alpha': 1e-05, 'max_iter': 500, 'tol': 0.01}
The validation RMSE is still relatively high, and the R-squared (R2) is negative, indicating that the model is not fitting well.
Suggestions:
You may consider trying a wider range of alpha values for stronger regularization.
You can experiment with different solver options, such as 'lbfgs' or 'elasticnet', to see if they perform better.
Decision Tree:
The best hyperparameters found: {'max_depth': 5, 'min_samples_leaf': 10, 'min_samples_split': 2}
The validation RMSE and R-squared (R2) indicate that the model is performing reasonably well, but there's room for improvement.
Suggestions:
You can try increasing the max_depth further to see if the model can capture more complex patterns in the data.
Experiment with a wider range of values for min_samples_split and min_samples_leaf, but be cautious not to over-regularize.
Random Forest:
The best hyperparameters found: {'max_depth': None, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 300}
The validation RMSE and R-squared (R2) indicate that the model is performing reasonably well.
Suggestions:
You might consider trying a wider range of n_estimators to see if increasing the number of trees leads to better performance.
Further tuning of max_depth can be done, but avoid setting it too high to prevent overfitting.
Gradient Boosting:
The best hyperparameters found: {'learning_rate': 0.005, 'max_depth': 3, 'n_estimators': 400}
The validation RMSE and R-squared (R2) indicate that the model is performing reasonably well, but there's room for improvement.
Suggestions:
Experiment with different learning rates to find the optimal balance between model convergence and avoiding overshooting.
You may consider increasing max_depth further to see if it captures more complex interactions, but be cautious about overfitting.
Remember to validate the model's performance on a separate test set or through cross-validation to ensure that the suggested parameter changes lead to improved generalization performance and avoid overfitting. It's an iterative process, so keep refining the parameters until you achieve the desired model performance on unseen data.
"""

#feedback 3

#alternative hyperparameters
"""
    sgd_hyperparameters = {
        'penalty': ['l2', 'l1','elasticnet'],
        'alpha': [0.1, 0.01, 0.001, 0.0001],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9 ],
        'max_iter': [500, 1000, 1500, 2000],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    }

    decision_tree_hyperparameters = {
    'max_depth': [10, 20, 50],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 3, 5, 7],
    'splitter': ['best', 'random'] 
    }
    random_forest_hyperparameters = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10,20,50],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 3, 5, 7]

    }
    gradient_boost_hyperparameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.001, 0.0001],
        'criterion': ['friedman_mse', 'squared_error'],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 3, 5, 7]
"""