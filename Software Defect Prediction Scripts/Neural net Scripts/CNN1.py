import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import minmax_scale
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("C:\Bakalauras\Software Defect Prediction Scripts\Datasets\cm1.csv", header=None)
dataset = dataframe.values
#print(dataset)
#Atskiriami duomenys nuo aprašo (angl. Label)
X=dataset[:,0:21].astype(float)
Y=dataset[:,21]
#print(X, Y)

#Užkoduojame aprašus integer reikšmėmis
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#print(Y, encoded_Y)

x_train = X.reshape(X.shape[0], X.shape[1], 1)
#print(x_train)

conv = Sequential()
conv.add(Conv1D(21, 4, input_shape = x_train.shape[1:3], activation = 'relu'))
conv.add(MaxPooling1D(2))
conv.add(Flatten())
conv.add(Dense(1, activation = 'sigmoid'))
sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
conv.fit(x_train, encoded_Y, batch_size = 150, epochs = 100, verbose = 0)

#Modelio įvertinimas su standatizuotais duomenimis
estimators = []
estimators.append(('standartize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(conv, epochs=150, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, x_train, encoded_Y, cv=kfold)
#results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

'''
print(numpy.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,21,1))
prediction = conv.predict(numpy.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,21,1))
print(prediction)
'''