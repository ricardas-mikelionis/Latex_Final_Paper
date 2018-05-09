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

#Bazinis modelis
def create_baseline():
	#modelio sukūrimas
	model = Sequential()
	model.add(Dense(21, input_dim=21, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	#modelio sukompiliavimas
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
#modelis su labiau sukoncentruotais duomenimis
def smaller_model():
	#modelio sukūrimas
	model = Sequential()
	model.add(Dense(15, input_dim=21, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	#modelio sukompiliavimas
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
#didenis modelis su dar vienu sluoksniu
def create_larger():
	#modelio sukūrimas
	model = Sequential()
	model.add(Dense(21, input_dim=21, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	#modelio sukompiliavimas
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
#Modelio įvertinimas su standatizuotais duomenimis
#estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
estimators = []
estimators.append(('standartize', StandardScaler()))
#estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
#estimators.append(('mlp', KerasClassifier(build_fn=smaller_model, epochs=100, batch_size=5, verbose=0)))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=150, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
#results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

'''
#Modelio eksportavimas į paveikslėlį
from keras.utils import plot_model
plot_model(create_baseline(), to_file='model.png')
'''