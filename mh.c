#include stdio.h
#include math.h
#include stdlib.h

/* C impementation of metropolis hastings for evaluating path integrals */

/* Action definitions */

float S_euclidean(float* path, struct params)
{

    /* TODO:  Return the value of the action for a path */
    return -1.0;

}

float delta_S(float* path, float delta, float path_index)
{

    /* TODO: Return value of change in action due to updating value at path_index by delta */
    return -1.0;

}

float metropolis_update(float* path, int n_sweeps)
{

	/* TODO: Return path after performing metropolis hastings updating for n_sweeps sweeps */
	return -1.0;
	
}

float burn_in(float* path, int n_burn)
{

	/* TODO: Return path after burning in for n_burn reps */
	return -1.0;
	
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