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
	int cell, samp;
	float df, bdif, p1;
	for(samp=0; samp < nsamps; samp++)
	{
		int sampstart = blk*N*nsamps + samp*N;
		/* need to include nsweeps */
		for(cell=0; cell < N; cell++)
		{

			for(j=0; j<=cell-1; j++)
			{
				bdif += beta[j*N-j*(j+1)/2 + cell];
			}
			for(j=cell+1; j<=N-1; j++)
			{
				bdif += beta[cell*N - cell*(cell+1)/2 +j];
			}
			
			df = -1.0*alpha[cell] - bdif;
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
		sample_covariance[cell*N] = 0.0;
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
	int nsamps_tot = samps_per_block*nblocks;
	int nsweeps = 10;

	/* Compute sizes of various data structures */
	size_t mean_size = N*sizeof(float);
	size_t cov_size = N*(N-1)/2*sizeof(float);
	size_t samples_size = N*nsamps_tot*sizeof(float);

	/* allocate result memory */
	float *alpha_res = (float*)malloc(mean_size);
	float *beta_res = (float*)malloc(cov_size);
	float *sample_mean_res = (float*)malloc(mean_size);

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

	/* generate random initial conditoins */
	int i;
	for(i=0; i<N; i++)
	{
		alpha_res[i] = 2*rand()-1;
	}
	for(i=0; i<N*(N-1)/2; i++)
	{
		beta_res[i]=2*rand()-1;
	}

	/* copy initial conditions over to device */
	cudaMemcpy(alpha, alpha_res, mean_size, cudaMemcpyHostToDevice);
	cudaMemcpy(beta, beta_res, cov_size, cudaMemcpyHostToDevice);

	/* sample */
	printf("Sampling...\n");
	init_rand<<<nblocks, 1>>>(d_state);
	gibbs_sample<<<nblocks, 1>>>(d_state, alpha, beta, samples, N, samps_per_block, nsweeps);
	printf("Finished.  nsamps=%d", nsamps_tot);
	printf("Computing Sample mean...");
	compute_sample_mean<<<N, 1>>>(samples, sample_mean, nsamps_tot, N);

	/*copy back sample mean*/
	cudaMemcpy(sample_mean_res, sample_mean, mean_size, cudaMemcpyDeviceToHost);

	/* Display*/
	for(i=0; i<N; i++)
	{
		printf("%f\n", sample_mean_res[i]);
	}


	/* free memory */
	free(alpha_res);
	free(beta_res);
	free(sample_mean_res);

	cudaFree(sample_mean);
	cudaFree(sample_covariance);
	cudaFree(samples);
	cudaFree(alpha);
	cudaFree(beta);
	cudaFree(d_state);

	return 0;
}
