import GPy
import numpy as np
from IPython.display import display
import plotly
#import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)
from matplotlib import pyplot as plt
#plotly.tools.set_credentials_file(username='bwei', api_key='ltnTdAO9V2Tb5fnZn3d4')
GPy.plotting.change_plotting_library('matplotlib')
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,kernel)
#m.optimize(messages=True)
display(m)
fig = m.plot(plot_density=False)
GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')

plt.show()





