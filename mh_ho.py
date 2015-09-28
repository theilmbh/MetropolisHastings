# Use Metropolis Hastings algorithm to evaluate path integrals for the 1-D Harmonic Oscillator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd
%pylab inline

# Define parameters

D = 1.5
Nt = 1000
T = 1.25
eps = 0.1
omega = 0.005/eps
Nmc = 20000
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
	
def S(path, g):
    
    return np.dot(path[1:-1].T, path[1:-1]) - g*np.dot(path[0:-1].T, path[1:])

def S_paths(paths, g):
    
    return np.diagonal(np.dot(paths[:, 1:-1].T, paths[:, 1:-1])) - np.diagonal(g*np.dot(paths[:, 0:-2].T, paths[:, 1:]))
  	
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
    
def generate_paths(g, Nmc):
    
    path = burn_in(init_path, 10, g)
    paths = np.zeros((Nmc, Nt))
    for mc_iter in range(Nmc):
        if np.mod(mc_iter, 50)==0:
            print(mc_iter)
        # run an iteration
        path = metropolis_update(path, g, Nsweeps)
        #measure
        paths[mc_iter, :] = np.squeeze(path)  
        
    return paths
    
# Here we actually measure things
#First, burn in
# Try to reproduce harmonic oscillator amplitude <0,T|0,0>
g = (1 - kappa)/(1+kappa)
paths = generate_paths(g, 4000)
print('done')
# Find how many were in the range 0 +/- 0.1
#at_origin = np.abs(paths) <= 0.5
#num_at_origin = np.sum(at_origin, 0)
#prob_at_origin = (1.0*num_at_origin) / Nmc

# Let's just average the paths and see what happens
avg_path = np.mean(paths, 0)
plt.figure()
plt.plot(avg_path)
