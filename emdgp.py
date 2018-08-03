import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
import GPy
from IPython.display import display

# imports done above
path = "/home/bwei/PycharmProjects/data lib/long wind"
windset = rd(path)
name = raw_input('the name of data set?')
realwindset = windset[name]
realwindset.shape = (len(realwindset),1)
x = np.linspace(1, len(realwindset), len(realwindset))
x.shape = (len(x), 1)

# read the data and reshape
