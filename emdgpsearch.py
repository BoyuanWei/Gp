import numpy as np
from readdata import read as rd
from emd import ee
import scipy.io as scio
import os
from termcolor import *
#@boyuan
#use this one to read the data and do EMD, then save every imfs as an individual .mat file in order to do the gp kernel functions search.

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

#save imfs as individual file.
if os.path.exists('./matdatafiles/'):
    print('folder exist.')
else:
    print('folder not exist, create one.')
    os.mkdir('./matdatafiles/')
for n in range(0, nimfs):
    datafolder = './matdatafiles/imf'+str(n+1)+'.mat'
    print(colored("saving..."+datafolder, 'green'))
    imfdata = imfs[n]
    imfdata.shape = (len(imfdata), 1)
    scio.savemat(datafolder, {'X': xtr, 'y': imfdata})

