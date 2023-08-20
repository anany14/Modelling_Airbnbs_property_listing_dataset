from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from tabular_data import load_airbnb
from typing import List, Tuple, Dict, Any
import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Split dataset into training, testing, and validation sets
def split_X_y(X: pd.DataFrame,y: pd.Series, train_size: float = 0.8) -> Tuple:
    """
    Split the dataset into training,testing and validation tests.
    
    Parameters
    ----------

        X : (pd.DataFrame) \n\t Features.
        y : (pd.Series) \n\t Labels.
        train_size : (float),defualt = 0.8 \n\t ratio of train size compared to the whole dataset

    Returns
    -------

        Tuple (X_train,y_train,X_test,y_test,X_val,y_val)    
        \nA tuple containing the split features and labels.
    """
    # Normalise features using MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X) 
    # encoding the labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Splitting training and testing data 80-20
    X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=train_size, random_state=24)
    # Splitting testing and validation data 50:50
    X_test,X_val,y_test,y_val = train_test_split(X_test,y_test, test_size=0.5, random_state=24)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training Label shape: {y_train.shape}")
    print("Number of samples in:")
    print(f"    Training:   {len(y_train)}")
    print(f"    Testing:    {len(y_test)}")
    print(f"    Validation:   {len(y_val)}\n")
    return (X_train,y_train,X_test,y_test, X_val,y_val)

# Train a logistic regression model and return performance metrics
def train_logistic_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,modelclass : type[LogisticRegression] = LogisticRegression)->Dict[str,float]:
    """
    Train a logistic regression classifier on the given dataset.
    
    Parameters
    ----------

        X_train : (pd.DataFrame) \nFeatures for training set.
        Y_train : (pd.Series) \nLabels for training set.
        X_test : (pd.DataFrame) \nFeatures for testing set.
        Y_test : (pd.Series) \nLabel for testing set.
        modelclass : (type), default = LogisticRegression) \nThe class for the logistic model.

    Return
    ------
        
        None

    """
    model = modelclass(max_iter=10000,random_state=24)
    # Fitting the Logistic Regression Model to the Training Set
    model.fit(X_train, y_train)

    # Predicting the Test set results
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculating accuracy,precision,recall and f1 scores
    train_accuracy_score = accuracy_score(y_train,y_train_pred)
    train_precision_score = precision_score(y_train,y_train_pred,average ='micro')
    train_recall_score = recall_score(y_train,y_train_pred,average ='micro')
    train_f1_score = f1_score(y_train,y_train_pred,average ='micro')

    print("Initial Logistic Regression Model parametrics:")
    print(f'Train Accuracy Score:\t {train_accuracy_score}')
    print(f'Train Precision Score:\t {train_precision_score}')
    print(f'Train Recall Score:\t {train_recall_score}')    
    print(f'Train F1 Score:\t \t {train_f1_score}')

    # Test set evaluation metrics
    test_accuracy_score = accuracy_score(y_test,y_test_pred)
    test_precision_score = precision_score(y_test,y_test_pred,average ='micro')
    test_recall_score = recall_score(y_test,y_test_pred,average ='micro')
    test_f1_score = f1_score(y_test,y_test_pred,average ='micro')

    print(f'Test Accuracy Score:\t {test_accuracy_score}')
    print(f'Test Precision Score:\t {test_precision_score}')
    print(f'Test Recall Score:\t {test_recall_score}')    
    print(f'Test F1 Score:\t \t {test_f1_score}')


# Tune hyperparameters of a model and return the best model and metrics
def tune_classification_model_hyperparameters(model_class:type,X_train:pd.DataFrame,y_train:pd.Series,
    X_val:pd.DataFrame,y_val:pd.Series,hyperparameters: Dict[str,float]) ->Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Perform a grid search over a range of hyperparameters for a given logistic_regression model using GridSearchCV.

    Parameters
    ----------

        model_class : (type) \nThe class for logistic regression.
        X_train : (pd.DataFrame) \nFeatures for training set.
        y_train : (pd.Series) \nLabels for training set.
        X_val : (pd.DataFrame) \nFeatures for validation set.
        y_val : (pd.Series) \nLabel for validation set.
        hyperparameters : (Dict[str,float]) \nDictionary containing all possible values for each hyperparameter

    Returns
    -------

        Tuple[Any, Dict[str, Any], Dict[str, float]] \n Tuple containing trained model,best parameters and performance metrics
    """
    model = model_class(random_state=24)
    gridsearch = GridSearchCV(estimator=model,param_grid=dict(hyperparameters),cv=5,scoring='accuracy')
    gridsearch.fit(X_train,y_train)

    # Training Data
    best_model = gridsearch.best_estimator_
    best_hyperparameters = gridsearch.best_params_
    train_accuracy_score = gridsearch.best_score_

    # Validation Data, Will use these scores to compare different models
    y_val_predict = best_model.predict(X_val)
    val_accuracy_score = accuracy_score(y_val,y_val_predict)
    val_precision_score = precision_score(y_val,y_val_predict,average ='micro')
    val_recall_score = recall_score(y_val,y_val_predict,average ='micro')
    val_f1_score = f1_score(y_val,y_val_predict,average ='micro')

    #making the dictionary performance metrics with scores
    performance_metrics = {"training_accuracy":train_accuracy_score,"validation_accuracy":val_accuracy_score,
    "validation_precision":val_precision_score,"validation_recall":val_recall_score,"validation_f1":val_f1_score}

    print(best_model)
    print(performance_metrics)
     
    return best_model,best_hyperparameters,performance_metrics

# Save trained model, hyperparameters, and performance metrics
def save_model(model:type, hyperparameters:dict, performance_metrics:dict, folder:str="models/classification/logistic_regression"):
    """
    A function that saves a model,it's hyperparameters and performance metrics in the designated folder.
    
    Parameters
    ----------

        model : (type) \n The model to be saved in the folder.
        hyperparameters : (dict) \n The model's hyperparameters.
        performance_metrics : (dict) \n The models performance metrics.
        folder : (str) \n The path to the designated folder passed as string.

    Returns
    -------

        None
    """
    # create folder if it doesn't exist
    os.makedirs(folder,exist_ok=True)

    #save model using joblib
    model_name = os.path.join(folder,'model.joblib')
    joblib.dump(model,model_name)
    
    #save hyperparameter values in a json file
    hyperparameters_name = os.path.join(folder,'hyperparameters.json')
    with open (hyperparameters_name,"w") as f:
        json.dump(hyperparameters,f,indent=4)

    # save performance metrics in a json file
    performance_metrics_name = os.path.join(folder, 'performance_metrics.json')
    with open(performance_metrics_name, 'w') as f:
        json.dump(performance_metrics, f, indent=4)

def plot_model_comparison(models: List[str], scores: List[float], title: str,file_path: str = None):
    """
    Plot a bar chart comparing different models based on their performance scores.

    Parameters
    ----------

        model : (list) \n
        score : (dict) \n
        title : (string) \n

    Return
    ------
        None
    """

    plt.figure(figsize=(10, 6))
    
    # Define colors for each model's bar
    colors = ['blue', 'green', 'orange', 'red']
    
    plt.bar(models, scores, color=colors)
    
    # Find the index of the best performing model
    best_model_idx = np.argmax(scores)

    # Retrieve the top score (maximum accuracy)
    top_score = max(scores)

    # Annotate the best performing model and top score
    plt.annotate(f"Best: {models[best_model_idx]}  (Top Score: {top_score:.4f})", 
                 xy=(best_model_idx, top_score), 
                 xytext=(5, -15), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='black'))

    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, max(scores) + 0.05)  # Adjust ylim for better visualization
    plt.xticks(rotation=45)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path,format='png')
        print(f"plot saved as {file_path}")
    else:
        plt.show()


# Evaluate different models, tune hyperparameters, and save results
def evaluate_all_models(X_train:pd.DataFrame,y_train:pd.Series,X_val:pd.DataFrame,y_val:pd.Series):
    """
    Evaluate different regression models using hyperparameter tuning and save results.

    Parameters:

        X_train : (pd.DataFrame) \nTraining features.
        y_train : (pd.Series) \nTraining labels.
        X_val : (pd.DataFrame) \nTraining features.
        y_val : (pd.Series) \nTraining labels.

    Returns:
        None
    """

    models_to_evaluate = [
    ("Logistic_Regression", LogisticRegression, {
        "penalty" : ['l2', None],
        "max_iter": [1000,10000,100000],
        "solver" : ['lbfgs','saga']
    }),
    ("DecisionTree_Classifier", DecisionTreeClassifier, {
        "criterion" : ["gini", "entropy", "log_loss"],
        "splitter" : ["best", "random"],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2,3,5],
        "min_samples_leaf": [1,2,5]
    }),
    ("RandomForest_Classifier", RandomForestClassifier, {
        "n_estimators": [100, 200, 300],
        "criterion" : ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 7],
        "min_samples_leaf": [1, 2, 5]
    }),
    ("GradientBoosting_Classifier", GradientBoostingClassifier, {
        'criterion': ['friedman_mse', 'squared_error'],
        "n_estimators": [50,100, 150],
        "max_depth": [3, 5, 10],
        "learning_rate": [0.01, 0.1,1,10]
    })
    ]
    for model_name,model,hyperparameters in models_to_evaluate:
        print(f"Checking {model_name}.....")
        best_model,best_hyperparameters,performance_metrics = tune_classification_model_hyperparameters(model,X_train,y_train,
            X_val,y_val,hyperparameters)

        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validated accuracy scores for {model_name}: {cv_scores}")
        print(f"Mean CV accuracy for {model_name}: {cv_scores.mean()}")

        #saving the model
        model_folder = f"models/classification/{model_name.lower()}"
        save_model(best_model,best_hyperparameters,performance_metrics,model_folder)
        print(f"Model Saved for {model_name}\n")

# Find the best model based on saved accuracy scores
def find_best_model(X_test,y_test):
    """
    Find the best model using the saved accuracy scores from previously tuned models.

    Parameters
    ----------
        None

    Returns
    -------
        None


    """
    best_model = None
    best_hyperparameters = {}
    best_accuracy = 0.0
    model_names = []
    model_scores = []

    models_to_evaluate = ["Logistic_Regression","DecisionTree_Classifier","RandomForest_Classifier","GradientBoosting_Classifier"]

    for model in models_to_evaluate:
        model_folder = f"models/classification/{model.lower()}"
        performance_metrics_path = os.path.join(model_folder,'performance_metrics.json')

        with open (performance_metrics_path,'r') as f:
            performance_metrics = json.load(f) 

        model_accuracy = performance_metrics.get("validation_accuracy",float)
        # modelling the model
        model_names.append(model)
        model_scores.append(model_accuracy)

        if model_accuracy > best_accuracy:
            best_model = joblib.load(os.path.join(model_folder,'model.joblib'))
            hyperparameters_path = os.path.join(model_folder,'hyperparameters.json')
            with open (hyperparameters_path,"r") as f:
                best_hyperparameters= json.load(f)
            best_accuracy = model_accuracy
            model_name_string = f"{model}"

    plot_model_comparison(model_names, model_scores, "Different Classification Models", "plots/classification/classification_models_comparison.png")

    y_test_pred = best_model.predict(X_test)
    test_accuracy_score = accuracy_score(y_test,y_test_pred)
    test_precision_score = precision_score(y_test,y_test_pred,average ='micro')
    test_recall_score = recall_score(y_test,y_test_pred,average ='micro')
    test_f1_score = f1_score(y_test,y_test_pred,average ='micro')

    print(f"\n{model_name_string} is the best model.")
    return best_model,best_hyperparameters,performance_metrics,test_accuracy_score,test_precision_score,test_recall_score,test_f1_score

# Main function to run the entire workflow
def main():
    """
    Main function to run all the function above and print the output.
    
    Parameters
    ----------

        None

    Returns
    -------

        None
    """
    # Load and preprocess data
    df = pd.read_csv('airbnb-property-listing/tabular_data/clean_listing.csv')
    X, y = load_airbnb(df, label='Category')
    X_train, y_train, X_test, y_test, X_val, y_val = split_X_y(X, y)
    
    # Train an initial logistic regression model and print its performance
    train_logistic_model(X_train, y_train, X_test, y_test)
    #print(f"\n Initial Logistic Regression Performance Metrics:\n\t Test Accuracy: {performance_metrics['test_accuracy']}\n\t Test Precision: {performance_metrics['test_precision']}\n\t Test Recall: {performance_metrics['test_recall']}\n\t Test f1: {performance_metrics['test_f1']}")
    X_train, y_train, X_test, y_test, X_val, y_val = split_X_y(X, y,train_size=0.65)
    # Evaluate various models using hyperparameter tuning and save results
    # evaluate_all_models(X_train, y_train, X_val, y_val)
    
    # Find the best model and print its hyperparameters and performance
    best_model, best_hyperparameters, performance_metrics, test_accuracy_score, test_precision_score, test_recall_score, test_f1_score = find_best_model(X_test,y_test)
    # print the best model's hyperparameters and performance metrics
    print(f"{best_model}")
    print(f"The best Hyperparameters are:\n\t {best_hyperparameters}")
    print(f"The best Performance Metrics are:\n\t {performance_metrics}")
    # print the best model's test performance metrics
    print(f'Test Accuracy:\t {test_accuracy_score}')
    print(f'Test Precision:\t {test_precision_score}')
    print(f'Test Recall:\t {test_recall_score}')    
    print(f'Test F1:\t {test_f1_score}')

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()