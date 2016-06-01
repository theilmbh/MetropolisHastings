#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

/* Initialize random number generators */
__global__ void init_rand(curandState *state)
{
	int idx = blockIdx.x;
	curand_init(1337, idx, 0, &state[idx]);
}

__global__ void gibbs_sample(curandState *state, float* alpha, float* beta, float* samples, 
							 int N, int nsamps, int nsweeps)
{

	int blk = blockIdx.x;
	int i, j;
	int cell;
	float s;
	float up, dp, df, p1;
	for(samp=0; samp < nsamps; samp++)
	{

		int sampstart = blk*N*nsamps + samp*N;
		/* need to include nsweeps */
		for(cell=0; cell < N; cell++)
		{
			samples[sampstart + cell] = 1.0;
			for(i = 0; i < N; i++)
			{
				up += -1.0*alpha[i]*samples[sampstart + i];
				for(j=0; j<i; j++)
				{
					up += -1.0*beta[(i + j)-1]*samples[sampstart+ i]*samples[sampstart + j];
				}
			}
			samples[sampstart + cell] = 0.0;
			for(i = 0; i < N; i++)
			{
				dp += -1.0*alpha[i]*samples[sampstart+ i];
				for(j=0; j<i; j++)
				{
					dp += -1.0*beta[(i + j)-1]*samples[sampstart + i]*samples[sampstart + j];
				}
			}
			df = up - dp;
			p1 =expf(df)/(1+expf(df));
			if(curand_uniform(&state[blk]) < p1)
			{
				samples[sampstart + cell] = 1.0;
			}
		}
	}
}

__global__ void compute_sample_mean(float* samples, float* sample_mean, int nsamps, int N)
{
	int cell = blockIdx.x;
	sample_mean[cell] = 0;
	for(int i = 0; i<nsamps; i++)
	{
		sample_mean[cell] += samples[cell + i*N];
	}
	sample_mean[cell] = sample_mean[cell] / nsamps;
}

__global__ void compute_sample_covariance(float* samples, float* sample_covariance, int N)
{
	int cell = blockIdx.x;
	for(int j=0; j<cell; j++)
	{
		sample_covariance[cell*N]
	}
}

__global__ void update_parameter_estimates(float* alpha, float* beta, float* sample_mean, float* sample_covariance, int N)
{

}

int main()
{

	int N = 20; /* Number of Neurons */
	int samps_per_block = 1024;
	int nblocks = 1024

	/* Compute sizes of various data structures */
	size_t mean_size = N*sizeof(float);
	size_t cov_size = N*(N-1)/2*sizeof(float);
	size_t samples_size = N*samps_per_block*nblocks*sizeof(float);

	/* allocate result memory */
	float *alpha_res = (float*)malloc(mean_size);
	float *beta_res = (float*)malloc(cov_size);

	/* Allocate device memory */
	float *sample_mean;
	float *sample_covariance;
	float *samples;
	float *alpha;
	float *beta;

	cudaMalloc(&sample_mean, mean_size);
	cudaMalloc(&sample_covariance, cov_size);
	cudaMalloc(&samples, samples_size);
	cudaMalloc(&alpha, mean_size);
	cudaMalloc(&beta, cov_size);

	curandState *d_state;
	cudaMalloc(&d_state, nblocks);





}
