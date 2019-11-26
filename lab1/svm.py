from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
target = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris.data, target, test_size=0.2, random_state=1)

svmm = svm.SVC(kernel='linear')
svmm.fit(x_train, y_train)
pred = svmm.predict(x_test)

for i in range(len(pred)):
    print('predicted_value: ', pred[i], ', true_value: ', y_test[i])
