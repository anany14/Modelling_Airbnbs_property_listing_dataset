from tabular_data import load_airbnb
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# loading the clean dataset as features and labels
df = pd.read_csv("C:/Users/Anany/OneDrive/Desktop/Github/AIcore/Modelling_Airbnb's_property_listing_dataset/airbnb-property-listing/tabular_data/clean_listing.csv")
X, y = load_airbnb(df, label="Price_Night")

#splitting dataset into training set and testing set 80:20 ratio and then 50:50 into testing and validation sets
#respectively for cross-validation purposes later on in model evaluation process
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=42)

print("Training data shape:", X_train.shape)
print("Training Label shape:", y_train.shape)

model = SGDRegressor(random_state=42) #creating an instance of linear regression algorithm
model.fit(X_train, y_train)#fitting our training set
y_pred = model.predict(X_test)


msevalue = mean_squared_error(y_test, y_pred,squared=False)
r2 = r2_score(y_test, y_pred)

print('msevalue: ',msevalue)
print('r2: ',r2)

#the outputs at this point suggest that there might be some issues with modelling
#extreme high values of mse and negative r-squared suggest that model is doing poorly and not fitting well

#Milestone 4 Task 2

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
print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Training R-squared (R2):", train_r2)
print("Test R-squared (R2):", test_r2)


#Milestone 4 Task 3