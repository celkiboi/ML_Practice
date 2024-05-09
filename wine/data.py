import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Zadatak 0.0.10

data = pd.read_csv('winequality-red.csv', sep=';')
# pod a
print(f'Mjerenje je provedeno na {data.shape[0]} vina')

#pod b
plt.hist(data['alcohol'], 6)
plt.title('Distribution of acohol content')
plt.xlabel('percentege of alcohol')
plt.ylabel('amount')
plt.show()

#pod c
low_quality = data[data['quality'] < 6]
high_quality = data[data['quality'] > 6]
print(f'Broj s kvalitetom < 6 {low_quality.shape[0]}')
print(f'Broj s kvalitetom > 6 {high_quality.shape[0]}')

#pod d
pd.set_option('display.max_columns', 12)
print(data.corr())

# 0.0.11
data['quality'] = data['quality'].apply(lambda x: 0 if x < 6 else 1)

input_variables = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
output_variables = ['quality']

X = data[input_variables]
y = data[output_variables]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.fit_transform(X_test)

# pod a
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

print(linearModel.coef_)

# pod b
y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test, y_test_p)
plt.title("Real values compared to predicted values")
plt.xlabel("Real values")
plt.ylabel("Predicted values")
plt.show()

# pod c
MAE = mean_absolute_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
RMSE = mean_squared_error(y_test, y_test_p, squared=False)
R_TWO_SCORE = r2_score(y_test, y_test_p)

print(f"MAE: {MAE}, MSE: {MSE}, MAPE: {MAPE}, RMSE: {RMSE}, R2 SCORE: {R_TWO_SCORE}")

# Zadatak 0.0.12
input_variables = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
output_variables = ['quality']

X = data[input_variables]
y = data[output_variables]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.fit_transform(X_test)

# pod a
input_shape = (11, )

model = keras.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Flatten())
model.add(layers.Dense(22, activation="relu"))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

# pod b
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#pod c
model.fit(X_train_n, y_train, epochs = 800, batch_size = 50, validation_split = 0.1)

#pod d
model.save('model.keras')
model = load_model('model.keras')

# pod e
score = model.evaluate(X_test_n, y_test, verbose = 0)
print(score)

# pod f
predictions = model.predict(X_test_n).round()
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions, labels=[0, 1]))
disp.plot()
plt.show()