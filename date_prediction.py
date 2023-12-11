#importing libraries
import pandas as pd
import numpy as np

#importing dataset
df = pd.read_excel('Day-wise planets degree and temperature.xlsx')

#changing date into numeric
df['Days'] = df['Date'].dt.dayofyear

#selecting features and target varible
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
#X = np.array(X)
#y = np.array(y)

#splitting into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#training random forest regressor model on trainig set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 350, random_state = 0)
regressor.fit(X_train,y_train)

#saving model in pickle file
from joblib import dump
pickle_out = open("regressor.pkl", mode = "wb") 
dump(regressor, pickle_out) 
pickle_out.close()

print("Model saved as pickle file named -> regressor.pkl")