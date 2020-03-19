import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import rand
import pylab 
import scipy.stats as stats

seed(0)

def Autocorrelation(t, landa):
    
    N ,  = np.shape(landa)
    
    landa_mean = np.mean(landa)
    
    auto_correlation = 0.
    
    for i in range (0, N - abs(t)):
        
        auto_correlation += (landa[i] - landa_mean) * (landa[i+abs(t)] - landa_mean)
        
    auto_correlation = (1/(N-abs(t))) * auto_correlation
    
    return auto_correlation

def AutocorrelationNormalized(t, landa):
    
    return Autocorrelation(t, landa) / Autocorrelation(0, landa)

def DataBlocking(landa):
    
    N ,  = np.shape(landa)
    
    Auto_1 = np.zeros(N)
    t1 = np.zeros(N)

    W = 0

    for i in range (0,N):
        Auto_1[i] = AutocorrelationNormalized(i, landa)
        t1[i] = i
        if (W == 0 and abs(Auto_1[i])<0.01):
            W = i

    tau_int = 0.
    for i in range (-W, W+1):
        tau_int += AutocorrelationNormalized(i, landa)

    l = int(tau_int) + 1
    landa_blocked = np.zeros(int(N/l))

    for i in range (0,int(N/l)):

        landa_blocked[i] = 1./float(l) * sum(landa[l*i: l*(i+1)])

    return landa_blocked

def Bootstrap(landa, R_parameter = 20, plotQQ = False):
    
    N, = np.shape(landa)
    R  = R_parameter * N
    
    landa_samples = np.zeros((R , N))
    boostrap_replications = np.zeros(R)
    
    for i in range (0 , R):
        
        for j in range (0 , N):
            
            k = int (rand(1) * N)
            landa_samples[i][j] = landa[k] 
        
        boostrap_replications[i] = np.mean(landa_samples[i])
    
    
    if (plotQQ == True):  
        
        plt.hist(boostrap_replications,50)
        plt.show()

        stats.probplot(boostrap_replications, dist="norm", plot=pylab)
        pylab.show()
    
    boostrap_mean = np.mean(boostrap_replications)
    std = 0.
    
    for i in range (0, R):
        std += (boostrap_replications[i] - boostrap_mean)**2
    
    std = np.sqrt(1./(R-1.)*std)

    return std

def SimpleBlockBootstrap(landa, cut = 0.1, plotQQ = False):
    
    L , = np.shape(landa)

    landa_cuted = landa[int(cut * L) :]
    
    landa_mean = np.mean(landa_cuted)
    
    landa_blocked = DataBlocking(landa_cuted)
    
    return landa_mean, Bootstrap(landa_blocked, plotQQ = plotQQ)