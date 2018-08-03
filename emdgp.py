import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
import GPy
from IPython.display import display
from emd import ee

# imports done above
path = "/home/bwei/PycharmProjects/data lib/long wind"
windset = rd(path)
name = raw_input('the name of data set?')
realwindset = windset[name]
realwindset.shape = (len(realwindset),1)
x = np.linspace(1, len(realwindset), len(realwindset))
x.shape = (len(x), 1)

# read the data and reshape above
days = raw_input('how mand days are used as training set?(less than 7)')
days = int(days)
trainset = realwindset[0:288*days]
testset = realwindset[288*days:len(realwindset)]

# generated training and test sets

xtr = np.linspace(1, len(trainset), len(trainset))
xte = np.linspace(1, len(testset), len(testset))
imfs = ee(trainset, 1)
#trainset decomposed

#start the GP:




