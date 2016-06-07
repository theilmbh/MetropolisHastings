#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

/* Initialize random number generators */
__global__ void init_rand(curandState *state, int sd)
{
	int idx = blockIdx.x;
	curand_init(1337, idx*sd, 0, &state[idx]);
}

__global__ void gibbs_sample(curandState *state, float* alpha, float* beta, float* samples, 
							 int N, int nsamps, int nsweeps)
{

	int blk = blockIdx.x;
	int i, j;
	int cell, samp, sweep;
	float df=0.0;
	float bdif=0.0;
	float p1=0.0;
	for(samp=0; samp < nsamps; samp++)
	{
		int sampstart = blk*N*nsamps + samp*N;
		for(sweep=0; sweep<nsweeps; sweep++)
		{
			for(cell=0; cell < N; cell++)
			{
				bdif=0.0;
				df = 0.0;
				p1=0.0;
				for(j=0; j<=cell-1; j++)
				{
					bdif += beta[j*(N)-j*(j+1)/2+i]*samples[sampstart+j];
				}
				for(j=cell; j<=N-1; j++)
				{
					bdif += beta[cell*(N)-cell*(cell+1)/2 +j]*samples[sampstart+j];
				}
				df = -1.0*alpha[cell] - bdif;
				p1 =expf(df)/(1+expf(df));
				if(curand_uniform(&state[blk]) < p1)
				{
					samples[sampstart + cell] = 1.0;
				}
				else
				{
					samples[sampstart + cell] = 0.0;
				}
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

__global__ void compute_sample_covariance(float* samples, float* sample_covariance, int nsamps, int N)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	sample_covariance[i*N +j] = 0.0;
	for(int samp = 0; samp<nsamps; samp++)
	{
		sample_covariance[i*N+j] += samples[samp*N+i]*samples[samp*N+j];
	}
	sample_covariance[i*N+j] = sample_covariance[i*N+j]/nsamps;

}


__global__ void update_alpha_estimate(float* alpha, float* sample_mean, float* data_mean, float eta, int N)
{

	int i = blockIdx.x;
	{
		alpha[i] += eta*(sample_mean[i] - data_mean[i]);
	}


}
__global__ void update_beta_estimate(float* beta, float* sample_covariance, float* data_covariance, float eta, int N)
{

	int i = blockIdx.x;
	{
		alpha[i] += eta*(sample_mean[i] - data_mean[i]);
	}


}


int write_paths_to_csv(char* fname, float* paths, int n_paths, int nt, char* mode)
{
    
    FILE *outfile;
    outfile = fopen(fname, mode);
    
    int i, j;
    if(outfile != NULL)
    {
        for(i = 0; i<n_paths; i++)
        {
            for(j = 0; j<nt; j++)
            {
                fprintf(outfile, "%f,", paths[i*nt + j]);
            }
            fprintf(outfile, "\n");
        }
    }
    fclose(outfile);
    return 0;
}

int main()
{

	int N = 20; /* Number of Neurons */
	int samps_per_block = 1024;
	int nblocks = 1024;
	int nsamps_tot = samps_per_block*nblocks;
	int nsweeps = 10;

	/* Compute sizes of various data structures */
	size_t mean_size = N*sizeof(float);
	size_t cov_size = (N*(N+1)/2)*sizeof(float);
	size_t full_cov_size = N*N*sizeof(float);
	size_t samples_size = N*nsamps_tot*sizeof(float);
	
	/* allocate result memory */
	float *alpha_res = (float*)malloc(mean_size);
	float *beta_res = (float*)malloc(cov_size);
	float *sample_mean_res = (float*)malloc(mean_size);
	float *sample_covariance_res = (float*)malloc(full_cov_size);
	float *samples_res = (float*)malloc(samples_size);

	/* Allocate device memory */
	float *sample_mean;
	float *sample_covariance;
	float *samples;
	float *alpha;
	float *beta;

	dim3 cov_block(N, N);

	cudaMalloc(&sample_mean, mean_size);
	cudaMalloc(&sample_covariance, full_cov_size);
	cudaMalloc(&samples, samples_size);
	cudaMalloc(&alpha, mean_size);
	cudaMalloc(&beta, cov_size);

	curandState *d_state;
	cudaMalloc(&d_state, nblocks);

	/* generate random initial conditoins */
	srand(time(NULL));
	int i;
	for(i=0; i<N; i++)
	{
		alpha_res[i] = (2*((float)rand()/(float)RAND_MAX) - 1);
		/*printf("%f\n", alpha_res[i]);*/
	}
	for(i=0; i<(N*(N+1)/2); i++)
	{
		beta_res[i]= 0.1*(2*((float)rand()/(float)RAND_MAX) - 1);
	}

	/* copy initial conditions over to device */
	cudaMemcpy(alpha, alpha_res, mean_size, cudaMemcpyHostToDevice);
	cudaMemcpy(beta, beta_res, cov_size, cudaMemcpyHostToDevice);

	/* sample */
	printf("Sampling...\n");
	init_rand<<<nblocks, 1>>>(d_state, time(NULL));
	gibbs_sample<<<nblocks, 1>>>(d_state, alpha, beta, samples, N, samps_per_block, nsweeps);
	printf("Finished.  nsamps=%d\n", nsamps_tot);
	printf("Computing Sample mean...\n");
	compute_sample_mean<<<N, 1>>>(samples, sample_mean, nsamps_tot, N);
	compute_sample_covariance<<<N, N>>>(samples, sample_covariance, nsamps_tot, N);
	/*copy back sample  mean*/
	cudaMemcpy(sample_mean_res, sample_mean, mean_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(samples_res, samples, samples_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_covariance_res, sample_covariance, full_cov_size, cudaMemcpyDeviceToHost);

	/* Display*/
	for(i=0; i<N; i++)
	{
		printf("%f\n", sample_mean_res[i]);
	}
	write_paths_to_csv("maxent_samples.csv", samples_res, nsamps_tot, N, "w");
	write_paths_to_csv("maxent_cov.csv", sample_covariance_res, N, N, "w");


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
