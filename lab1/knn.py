from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
target = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris.data, target, test_size=0.2, random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

for i in range(len(pred)):
    print('predicted_value: ', pred[i], ', true_value: ', y_test[i])
