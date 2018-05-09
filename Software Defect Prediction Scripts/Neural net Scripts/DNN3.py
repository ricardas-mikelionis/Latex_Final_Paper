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

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, encoded_Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

prediction = loaded_model.predict(numpy.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,21))
print(prediction)