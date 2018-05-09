import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import model_from_json
import os

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

#Modelis su trimis sluoksniais
def create_model():
	#modelio sukūrimas
	model = Sequential()
	model.add(Dense(21, input_dim=21, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	#modelio sukompiliavimas
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

predictor_model = create_model()
predictor_model.fit(X, encoded_Y, epochs=150, batch_size=10, verbose=0)

#modelio serializavimas į JSON failą
model_json = predictor_model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
#serializuojami "weights" į HDF5
predictor_model.save_weights("model.h5")
print("saved model to disk")
	
#Modelio įvertinimas su standatizuotais duomenimis
estimators = []
estimators.append(('standartize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(create_model, epochs=150, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
#results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

prediction = create_model().predict(numpy.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,21))
print(prediction)

'''
#Modelio eksportavimas į paveikslėlį
from keras.utils import plot_model
plot_model(create_baseline(), to_file='model.png')
'''