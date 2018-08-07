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
nimfs = len(imfs) #see how manys imfs are got from the emd

#reshape
xtr.shape = (len(xtr), 1)
#start the GP:
m=[]
GPy.plotting.change_plotting_library('matplotlib')

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

for n in np.arange(nimfs):
    imf = imfs[n]
    imf.shape = (len(imf), 1)
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m.append(GPy.models.GPRegression(xtr, imf, kernel))
    m[n].optimize(messages=True)
    display(m[n])
    fig = m[n].plot(plot_density=False, figsize=(14, 6), dpi=300)
    GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')
    plt.show()