# Use Metropolis Hastings algorithm to evaluate path integrals for the 1-D Harmonic Oscillator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd
%pylab inline

# Define parameters

D = 1.5
omega = 250.0
eps = 0.25 / omega
T = 1.0
Nt = int(np.floor(4*omega*T))
Nmc = 400
Nsweeps = 10
kappa = 0.25*(eps**2)*(omega**2)

init_path = np.zeros((Nt, 1))

# Define Euclidean action

def SE_ho(path, m, omega, eps):

    # estimate derivative
    dqdt = (path[1:] - path[0:-1]) / eps
    
    K = m*np.dot(dqdt, dqdt)/2.0
    V = m*omega**2 * np.dot(path, path)/2.0
    return -1.0*eps*(K+V)
    
def delta_SE_ho(path, delta, j, g):

	return delta*(delta + 2*path[j] - g*(path[j-1] + path[j+1]))
	
def metropolis_update(path, g, Nsweeps):

    for sweep in range(Nsweeps):
        for tstep in range(Nt-1):
            shift = 2.0*D*(np.random.rand() - 0.5)
            
            deltaS = delta_SE_ho(path, shift, tstep, g)
            accept_prob = np.min([1, np.exp(-1.0*deltaS)])
            if np.random.rand() <= accept_prob:
                path[tstep] = path[tstep]+shift
    return path
        
        
def burn_in(path, Nburn, g):

    for burn in range(Nburn):
    	print(burn)
        path = metropolis_update(path, g, Nsweeps)
        
    return path
    
def main():

    # Here we actually measure things
    #First, burn in
    g = (1 - kappa)/(1+kappa)
    path = burn_in(init_path, 10, g)
    qfinals = np.zeros((Nmc, 1))
    for mc_iter in range(Nmc):
        print(mc_iter)
        # run an iteration
        path = metropolis_update(path, g, Nsweeps)
        #measure
        qfinals[mc_iter] = path[-2]
    qfinals_distrib = pd.DataFrame(qfinals)
    plt.figure()
    qfinals_distrib.plot(kind='hist')
        
if __name__ == '__main__':
    main()
        