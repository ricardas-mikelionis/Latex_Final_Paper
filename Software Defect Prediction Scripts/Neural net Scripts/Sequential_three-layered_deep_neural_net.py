import numpy as np
import pandas as pd
#import matplotlib.pyplot as pl
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.utils import plot_model

#duomenų nuskaitymas
data = pd.read_csv("/Users/ricardas_mikelionis_mac/Documents/GitHub/Latex_Final_Paper/Software Defect Prediction Scripts/Datasets/cm1_2.csv")
seed = 5
np.random.seed(seed)

#print(data.head(2))
#data.info()

#prediction_var=['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']
#prediction_var=['loc', 'v(g)', 'ev(g)', 'iv(g)', 'v', 'd', 'i', 'e', 'b', 't']
prediction_var=['loc', 'iv(g)', 'n', 'v', 'i', 'b', 'lOComment', 'uniq_Op', 'uniq_Opnd', 'total_Op']
X = data[prediction_var].values
Y = data.defects.values

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


#funkcija gražinanti neuroninio tinklo modelį
def create_model():
	#modelio sukūrimas
	model = Sequential()
	model.add(Dense(6,input_dim=10, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, kernel_initializer='normal', activation='hard_sigmoid'))
	#Sukompiliavimas. Naudojant logarithmic loss function bei rmsprop gradient optimizer.
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
'''	
#Vertiname modelį standartizuotais duomenimis. 
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=3, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''
model = create_model()
model.fit(X, encoded_Y, epochs=100, batch_size=50, verbose=1, validation_split=0.3)

#plot_model(model, to_file='model.png', show_shapes=True)
