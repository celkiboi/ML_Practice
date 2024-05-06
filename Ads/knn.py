import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def do_KNN(n_neighbors: int):
    KNN_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNN_model.fit(X_train_n, y_train)

    y_train_p = KNN_model.predict(X_train_n)
    y_test_p = KNN_model.predict(X_test_n)

    print(f'for {n_neighbors} neighbors: ')
    print(f'Accuracy train: {accuracy_score(y_train, y_train_p)}')
    print(f'Accuracy test: {accuracy_score(y_test, y_test_p)}')
    print()


data = pd.read_csv("Social_Network_Ads.csv")

data.hist()
plt.show()

X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.fit_transform(X_test)


for i in range(1, 15, 2):
    do_KNN(i)