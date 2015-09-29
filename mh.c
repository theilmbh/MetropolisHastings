#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* C implementation of metropolis hastings for evaluating path integrals */

/* Action definitions */

float S_euclidean(float paths[], float g, int n_paths, int nt)
{

    /* Returns the value of the action for a path */
    int i, j;
    float A = 0;
    float B = 0;
    
    /* np.diagonal(np.dot(paths[:, 1:-1].T, paths[:, 1:-1])) - np.diagonal(g*np.dot(paths[:, 0:-2].T, paths[:, 1:])) */
    for(i = 0; i < n_paths; i++)
    {
    	for(j = 0; j < nt-1; j++)
    	{
    		A += paths[i][j+1]*paths[i][j+1];
    		B += g*paths[i][j]*paths[i][j+1];
    	}
    } 
    
    return A+B;

}

float delta_S(float* path, float g, float delta, int path_index)
{

    /* Returns value of change in action due to updating value at path_index by delta */
    return delta*(delta + 2*path[path_index] - g*(path[path_index-1] + path[path_index+1]));

}

float metropolis_update(float* path, int n_sweeps)
{

	/* TODO: Return path after performing metropolis hastings updating for n_sweeps sweeps */
	return -1.0;
	
}

float burn_in(float* path, int n_burn)
{

	/* Burn in path for n_burn reps */
	int burn;
	int n_burn_sweeps = 100;
	for(burn = 0; burn < n_burn; burn++)
	{
		metropolis_update(path, n_burn_sweeps);
	}
	return 0;
	
}

float* get_sample_paths(int n_paths)
{

	/* TODO: Use the metropolis hastings algorithm to generate n_paths samples according to the action */
	float x = -1;
	return &x;
	
}

int main()
{

	return 0;

}