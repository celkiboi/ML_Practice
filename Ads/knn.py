import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)

def do_KNN(n_neighbors: int):
    KNN_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNN_model.fit(X_train_n, y_train)

    y_train_p = KNN_model.predict(X_train_n)
    y_test_p = KNN_model.predict(X_test_n)

    print(f'for {n_neighbors} neighbors: ')
    print(f'Accuracy train: {accuracy_score(y_train, y_train_p)}')
    print(f'Accuracy test: {accuracy_score(y_test, y_test_p)}')
    print()

    plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
    plt.tight_layout()
    plt.show()


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