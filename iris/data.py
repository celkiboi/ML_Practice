import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

iris = datasets.load_iris()
#print(iris['target_names'])
#print(iris['target'])
#print(iris['feature_names'])
#print(iris['data'])
#print(iris)

iris_dict = {
    'target': iris['target']
}

index = -1
for name in iris['feature_names']:
    index += 1
    iris_dict[name] = iris['data'][:, index]

data = pd.DataFrame(iris_dict)
data['target'] = iris['target_names'][data['target']]

#print(data)

# 0.0.7

# pod a)
virginica = data[data['target'] == 'virginica']
setosa = data[data['target'] == 'setosa']

plt.scatter(virginica['petal length (cm)'], virginica['sepal length (cm)'], c='green', label='virginica')
plt.scatter(setosa['petal length (cm)'], setosa['sepal length (cm)'], c='gray', label='setosa')
plt.title('petal vs sepal length')
plt.xlabel('petal length in cm')
plt.ylabel('sepal length in cm')
plt.legend()
plt.show()

# pod b)
versicolour = data[data['target'] == 'versicolour']
categories = iris['target_names']
values = []

for name in categories:
    biggest = data[data['target'] == name]['sepal width (cm)'].max()
    values.append(biggest)

plt.bar(categories, values)
plt.title('Highest recorded sepal width')
plt.ylabel('Sepal width (cm)')
plt.show()

#pod c)
average_sepal_width_setosa = setosa['sepal width (cm)'].mean()
above_average_sepal_width_setosa = setosa[setosa['sepal width (cm)'] > average_sepal_width_setosa]
print(f'number of setosa with above average sepal width: {above_average_sepal_width_setosa.shape[0]}')

# 0.0.8


# 0.0.9
input_variables = iris['feature_names']
output_variables = ['target']

X = data[input_variables]
y = data[output_variables]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)

# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)


# ohe = OneHotEncoder()
# y_train_encoded = ohe.fit_transform(y_train)
# y_test_encoded = ohe.transform(y_test)

y_train_encoded = pd.get_dummies(y_train)
y_test_encoded = pd.get_dummies(y_test)

#print(y_train_encoded)

# y_train_s = keras.utils.to_categorical(y_train_encoded, 3)
# y_test_s = keras.utils.to_categorical(y_test_encoded, 3)

# print(y_train_s)

input_shape = (4,)
model = keras.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Flatten())
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(7, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train,
            y_train_encoded,
            epochs = 450,
            batch_size = 7,
            validation_split = 0.1)

model.save('celic_model.keras')
model = load_model('celic_model.keras')

score = model.evaluate(X_test, y_test_encoded, verbose = 0)

predictions = model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(np.argmax(y_test_encoded, axis=1), np.argmax(predictions, axis=1)))
disp.plot()
plt.show()