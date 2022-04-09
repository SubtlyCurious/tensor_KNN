from os import sep
import sklearn
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

#Reading Data File
data = pd.read_csv('car.data', sep=',')
print(data.head())

#convert string to numerical data
le = preprocessing.LabelEncoder()

#turn into list and transform them into integer values
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

#prints [3 3 3 ... 1 1 1] transformed into integers
#print(buying)

#What I want to predict
predict = 'class'

#converts to one big list (zip -> creates tuple objects with attributes)
x = list(zip(buying, maint, door, persons, lug_boot, safety)) #features
y = list(cls) #labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
