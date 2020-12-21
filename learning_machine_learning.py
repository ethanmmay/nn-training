import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas
from sklearn.preprocessing import StandardScaler
from scipy import stats

scale = StandardScaler()

df = pandas.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])
predictedCO2 = regr.predict([scaled[0]])

print(predictedCO2)






