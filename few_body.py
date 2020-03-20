import numpy as np
from numpy.random import seed
from numpy.random import rand
from numpy.random import randn
from numpy.linalg import norm
seed(0)


def NewVector(R_m):
    L, = np.shape(R_m)
    
    if L/3 - int(L/3) != 0: print ("\nERROR: the size of the array does not match the L = 3N statement!\n")
        
    random_chain = rand(L+1)
    
    u = -np.log(np.prod(random_chain[1:L+1]))
    v = np.sqrt(1.-random_chain[0]**(2./(L-1.))) 
    
    R = u*v
    
    rho = randn(L)
    
    Omega = rho / norm(rho)
    
    return R_m + R*Omega



def SquaredWell(R_m, b):
    
    L, = np.shape(R_m)
    N = int(L/3) 
    
    suma = 0.
    
    for i in range (0 , N):
        for j in range (i+1, N):
            if (norm(R_m[3*i:3*(i+1)]-R_m[3*j:3*(j+1)]) < b):
                suma += 1
    
    return suma


def GaussianWell(R_m, b):
    
    L, = np.shape(R_m)
    N = int(L/3) 
    
    suma = 0.
    
    for i in range (0 , N):
        for j in range (i+1, N):
            suma +=  np.exp(-(1.435 * norm(R_m[3*i:3*(i+1)]-R_m[3*j:3*(j+1)])/b)**2)
    
    return suma



def ExponentialWell(R_m, b):
    
    L, = np.shape(R_m)
    N = int(L/3) 
    
    suma = 0.
    
    for i in range (0 , N):
        for j in range (i+1, N):
            suma += np.exp(-3.5412 * norm(R_m[3*i:3*(i+1)]-R_m[3*j:3*(j+1)])/b)
    
    return suma

def GreenFunctionMonteCarlo(N, Well, b_r, landa_0, Niter, Nthreshold, output = 0, R_as_output = False):

    R = np.zeros ((1,3*N))
    R_new = np.zeros ((1,3*N))
    landa = np.zeros(Niter)
    i2 = -1
    
    for i in range (0,Niter):

        size = np.shape(R)

        sum_W = 0.

        for m in range (0,size[0]):

            W = Well(R[m], b_r)

            sum_W += W

            mean_dist = landa_0 * W 

            integer_part = int(mean_dist)

            probability_dist =  mean_dist - integer_part

            if (rand(1) < probability_dist): e_m = integer_part + 1

            else: e_m = integer_part

            for k in range (0,e_m):

                #print ("m = ", m, "   k = ", k)

                if (i2 != i and k == 0): 

                    R_new = np.zeros ((1,3*N))
                    R_new[0] = NewVector(R[m])
                    i2 = i

                else: 
                    R_new = np.vstack ([R_new, NewVector(R[m])]) 

                    
        if (R_new[0].all() == 0): break
        
        landa[i] = size[0]/sum_W
        
        if (output == 1): print ("i = ", i+1, "  Size = ", size[0], "  Sum(W[i]) = ", sum_W, "  =>  landa[i] = "
                                 ,landa[i], end="\r")


        if (np.shape(R_new)[0] > Nthreshold or i > 10): landa_0 = landa[i]

        R = R_new

        R_new = np.zeros ((1,3*N))
    
    if (R_as_output == True):
        return R
    
    return landa