# Modelling_Airbnbs_property_listing_dataset


Welcome to the **Airbnb Model Evaluation Framework**! This project aims to create a versatile framework for systematically training, tuning, and evaluating machine learning models for a wide range of tasks, inspired by the challenges faced by the Airbnb team. Whether you're working with tabular, image, or text data, this framework will help you build effective models and streamline the evaluation process.

## Built With

This project leverages several essential frameworks and tools to achieve its goals:

- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis in Python.
- [Scikit-Learn](https://scikit-learn.org/) - Machine learning library in Python.
- [NumPy](https://numpy.org/) - Fundamental package for scientific computing with Python.
- [Joblib](https://joblib.readthedocs.io/) - Efficiently save and load Python objects.
- [JSON](https://www.json.org/) - For storing hyperparameters and performance metrics.
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) - Grid search for hyperparameter tuning.
- [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) - Feature scaling.

## Getting Started

To get started with the Airbnb Model Evaluation Framework, follow these steps:

### Prerequisites

Before you begin, ensure you have the following installed:

- Python (>=3.6)
- Jupyter Notebook (optional)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/anany14/Modelling_Airbnbs_property_listing_dataset.git
cd Modelling_Airbnbs_property_listing_dataset
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

Use the Airbnb Model Evaluation Framework to build and evaluate machine learning models. The provided python scripts demonstrate how to use the framework for regression and classification tasks. Feel free to modify the scripts or create your own to adapt the framework to your specific dataset and task.

1. Run the `tabular_data.py` script to download the dataset as a [PANDAS] DataFrame and then clean the data, remove unnecessary data, convert datatypes and get all the columns in the right format with the correct values.

2. Run the `modelling.py` script to build and evaluate regression models to predict the price per night of the Airbnb property listings using various models such as [SGDRegressor],[DecisionTreeRegressor],[RandomForestRegressor] and [GradientBoostingRegressor]

3. 

4. 

5. 


<!---

## Roadmap
We have some exciting plans for the future of this framework:

- Add support for additional data types and tasks.
- Enhance the visualization of results.
- Improve documentation with more usage examples.
- Introduce multi-language support.
- Include more configurable neural network architectures.
--->

## Acknowledgments

We'd like to acknowledge the following resources that have been instrumental in building this framework:

- [Scikit-Learn](https://scikit-learn.org/) - Providing a powerful and flexible machine learning library.
- [Pandas](https://pandas.pydata.org/) - Enabling efficient data manipulation and analysis.
- [NumPy](https://numpy.org/) - Essential for scientific computing in Python.
- [Joblib](https://joblib.readthedocs.io/) - Helping with efficient object serialization.
- [GitHub](https://github.com/) - Hosting and version control for collaborative development.

## Introduction

This project aims to build a systematic framework for training, tuning, and evaluating machine learning models on various tasks, inspired by the challenges tackled by the Airbnb team. The framework will be designed to handle different types of data, including tabular, image, and text data. The goal is to create a flexible and reusable system that can be applied to any dataset.

## Milestones
1. Set up the environment: Prepare the development environment to start building the framework.

2. Data preparation: Understand the structure of the Airbnb dataset and perform data cleaning and preprocessing.

3. Create a regression model: Build machine learning models that predict the price of the Airbnb listing per night and evaluate their performance.

4. Create a classification model: Develop classification models for specific tasks and assess their effectiveness.

5. Create a configurable neural network: Utilize a neural network to predict the nightly listing price using the numerical data from the tabular dataset.

6. Reuse the framework for another use-case: Test the flexibility of the framework by applying it to a different dataset, ensuring that it can handle various data types.

## Data Cleaning (tabular_data.py)

The `tabular_data.py` module contains functions for cleaning and preprocessing the Airbnb property listing dataset. The key steps include:

1. **Removing Rows with Missing Ratings:** The `remove_rows_with_missing_columns` function removes rows with missing values in specific rating columns (e.g., cleanliness, accuracy, communication, location, check-in, and value ratings).

2. **Fixing Problematic Rows:** The `fix_problematic_rows` function manually fixes specific rows with shifting issues in column values, ensuring proper alignment.

3. **Combining Description and Amenities:** The `combine_description_settings` function processes the description and amenities columns by cleaning and combining list items into a single string.

4. **Setting Default Feature Values:** The `set_default_feature_values` function fills empty entries in the guests, beds, bathrooms, and bedrooms columns with default values.

5. **Converting Data Types:** The `convert_dtypes_and_optimise_df` function converts specific columns to appropriate data types, optimizes the DataFrame, and drops unnecessary columns.

6. **Cleaning Tabular Data:** The `clean_tabular_data` function applies a series of data cleaning steps, combining the above functions to obtain a cleaned DataFrame.

## Regression Modeling (modelling.py)

The `modelling.py` module focuses on building and evaluating regression models to predict the price per night of the Airbnb property listings. The key steps include:

1. **Data Splitting:** The `split_X_y` function splits the dataset into training, testing, and validation sets, ensuring that the data is ready for model training and evaluation.

2. **Initial Model Training:** The `train_regression_model` function trains a regression model (default is Stochastic Gradient Descent - SGDRegressor) and prints its initial performance on the training and test sets.

3. **Custom Hyperparameter Tuning:** The `custom_tune_regression_model_hyperparameters` function performs a grid search over a range of hyperparameter values for a given regression model. It returns the best model, best hyperparameters, and performance metrics (validation RMSE, test RMSE).

4. **Hyperparameter Tuning with GridSearchCV:** The `tune_regression_model_hyperparameters` function uses GridSearchCV for hyperparameter tuning, allowing us to explore a wider range of hyperparameter values for different regression models.

5. **Model Evaluation and Selection:** The `evaluate_all_models` function evaluates multiple regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) by tuning their hyperparameters. The best model is selected based on validation RMSE, and the trained models are saved along with their performance metrics.

6. **Finding the Best Model:** The `find_best_model` function identifies the best-performing model based on the saved validation RMSE values from the earlier model evaluations.

## Model Selection and Metrics

In the file `modelling.py`, we evaluated several regression models:

1. **Linear Regression (SGDRegressor)**: We used Stochastic Gradient Descent as a baseline regression model.

2. **Decision Tree Regressor**: A decision tree-based regression model that can capture non-linear relationships.

3. **Random Forest Regressor**: An ensemble model combining multiple decision trees for improved predictive performance.

4. **Gradient Boosting Regressor**: A boosting algorithm that combines weak learners into a strong predictive model.

## Model Performance

The best model based on the validation RMSE is the chosen model for making predictions on new data. The key metrics used for evaluation are:

1. **Root Mean Squared Error (RMSE)**: A measure of the average deviation between the predicted and actual values. Lower RMSE indicates better model performance.

2. **R-squared (R2) Score**: A measure of how well the model explains the variance in the target variable. Higher R2 score indicates a better fit to the data.

## Model Performance Metrics 

**Linear Regression (SGDRegressor)**:
    - best hyperparameters: {'alpha': 1e-05, 'max_iter': 1000, 'tol': 0.001}
    - gridsearch_rmse: 97.7362631514105
    - validation_rmse: 127.4768534817222
    - validation_r2: -0.23837397130168658

**Decision Tree Regressor**:
    - best hyperparameters: {'max_depth': 5, 'min_samples_leaf': 10, 'min_samples_split': 2}
    - gridsearch_rmse: 99.83932831904127
    - validation_rmse: 104.07966226324588
    - validation_r2: 0.174492747038261

**Random Forest Regressor**:
    - best hyperparameters: {'max_depth': None, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 300}
    - gridsearch_rmse: 93.53765214999888
    - validation_rmse: 101.63984683370312
    - validation_r2: 0.212741880233853

**Gradient Boosting Regressor**:
    - best hyperparameters: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 300}
    - gridsearch_rmse: 102.7416140657607
    - validation_rmse: 107.80684569675182
    - validation_r2: 0.11430983256579175

## Further Experiments

While we have explored a variety of regression models and performed hyperparameter tuning, there are several additional experiments we could consider:

1. **Feature Engineering**: We can experiment with creating new features based on domain knowledge or feature interactions to potentially improve model performance.

2. **Advanced Ensemble Models**: We can explore more advanced ensemble methods such as XGBoost and LightGBM to see if they provide further improvements.

3. **Cross-Validation Strategies**: We used a simple train-validation-test split, but we can experiment with more advanced cross-validation strategies to robustly evaluate model performance.

4. **Fine-Tuning Hyperparameters**: We can perform more exhaustive grid searches or use Bayesian optimization to fine-tune hyperparameters and achieve even better model performance.

5. **Handling Outliers**: Exploring techniques to handle outliers in the data may improve the robustness of our models.


















## Contributing

Contributions to this project are highly appreciated. If you have suggestions or improvements, please follow these steps:

1. Fork the project.
2. Create a new branch: `git checkout -b feature/awesome-feature`.
3. Commit your changes: `git commit -m 'Add some awesome feature'`.
4. Push the branch: `git push origin feature/awesome-feature`.
5. Open a pull request with the tag "enhancement."

Your contributions will help make this framework even more useful to the community.

## License

Distributed under the MIT License. See [LICENSE.txt](LICENSE.txt) for more information.

## Contact

For any questions or inquiries, please feel free to reach out:

Anany Tripathi - ananytripathi10@gmail.com

Project Link: [https://github.com/anany14/Modelling_Airbnbs_property_listing_dataset]
