from numpy import genfromtxt

my_data = genfromtxt('/Users/ricardas_mikelionis_mac/Documents/GitHub/Latex_Final_Paper/Software Defect Prediction Scripts/Datasets/pc1.txt', delimiter=',')
features = [row[:-1] for row in my_data]
labels = [row[-1] for row in my_data]

#testdata = genfromtxt('C:\Bakalauras\Software Defect Prediction Scripts\Datasets\cm1.csv', delimiter=',')
#t_features = [row[:-1] for row in testdata]
#t_labels = [row[-1] for row in testdata]

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test =  train_test_split(features, labels, test_size = .5)

'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

'''
from sklearn import tree
classifier = tree.ExtraTreeClassifier()


classifier.fit(features, labels)
'''
from sklearn.externals.six import StringIO
import pydotplus as pydot
dot_data = StringIO()
KNeighborsClassifier.export_graphviz(classifier, out_file=dot_data, feature_names = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t", "lOCode", "lOComment", "lOblank", "lOCodeAndComment", "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"], class_names = ["true", "false"], filled = True, rounded= True, impurity= False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("tree.pdf")
'''

predictions = classifier.predict(features_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, predictions))
