import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
import GPy
from IPython.display import display
from GPy.kern import LinearSlopeBasisFuncKernel, DomainKernel, ChangePointBasisFuncKernel
import pandas as pd

path = "/home/bwei/PycharmProjects/data lib/long wind"
windset = rd(path)
name = raw_input('the name of data set?')
realwindset = windset[name]/10
realwindset.shape = (len(realwindset),1)
x = np.linspace(1, len(realwindset), len(realwindset))
x.shape = (len(x), 1)
GPy.plotting.change_plotting_library('matplotlib')
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)


#k = GPy.kern.Matern32(1, .3)
#Kf = k.K(x)
#k_per = GPy.kern.PeriodicMatern32(1, variance=100, period=1)
#k_per.period.fix()
#k_dom = DomainKernel(1, 1., 5.)
#k_perdom = k_per * k_dom
#Kpd = k_perdom.K(x)


#starts, stops = np.arange(0, 10, 3), np.arange(1, 11, 3)

#k = (GPy.kern.Bias(1)
   #  + GPy.kern.Matern52(1)
 #    + LinearSlopeBasisFuncKernel(1, ARD=1, start=starts, stop=stops, variance=.1, name='linear_slopes')
  #   + k_perdom.copy()
   # )

#k.randomize()
#

m = GPy.models.GPRegression(x, realwindset, kernel)
m.optimize(messages=True)
display(m)
fig = m.plot(plot_density=False)
GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')

plt.show()