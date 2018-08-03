from PyEMD import EMD, EEMD
import matplotlib.pyplot as plt
import numpy as np
from termcolor import *

def normal(data, drawflag):
    data.shape = (len(data),)
    x = np.linspace(1, len(data), len(data))
    print(colored("decomposing - normal emd mode...", 'green'))
    imfs = EMD().emd(data)
    if drawflag == 1:
        size = imfs.shape
        plt.figure(figsize=(20, 18))
        for loop in range(1, size[0]+1):
            plt.subplot(size[0], 1, loop)
            plt.plot(x, imfs[loop-1])
            plt.title(loop)
        plt.show()
    return imfs


def ee(data, drawflag):
    data.shape = (len(data),)
    x = np.linspace(1, len(data), len(data))
    eemd = EEMD()
    # Say we want detect extrema using parabolic method
    emd = eemd.EMD
    emd.extrema_detection = "parabol"
    print(colored("decomposing - EEmd mode...", 'green'))
    imfs = eemd.eemd(data, x)
    if drawflag == 1:
        size = imfs.shape
        plt.figure(figsize=(20, 18))
        for loop in range(1, size[0]+1):
            plt.subplot(size[0], 1, loop)
            plt.plot(x, imfs[loop-1])
            plt.title(loop)
        plt.show()
    return imfs