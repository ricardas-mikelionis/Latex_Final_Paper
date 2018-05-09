from numpy import genfromtxt

my_data = genfromtxt('C:\Bakalauras\Software Defect Prediction Scripts\Datasets\pc1.txt', delimiter=';')
features = [row[:-1] for row in my_data]
labels = [row[-1] for row in my_data]

#testdata = genfromtxt('C:\Bakalauras\Software Defect Prediction Scripts\Datasets\cm1.csv', delimiter=';')
#t_features = [row[:-1] for row in testdata]
#t_labels = [row[-1] for row in testdata]

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test =  train_test_split(features, labels, test_size = .5)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
'''

from sklearn import tree
classifier = tree.DecisionTreeClassifier()
'''
classifier.fit(features, labels)

predictions = classifier.predict(features_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, predictions))