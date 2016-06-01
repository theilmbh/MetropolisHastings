#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

__global__ void init_rand(curandState *state)
{
	int idx = blockIdx.x;
	curand_init(1337, idx, 0, &state[idx]);
}

__global__ void mh_update(curandState *state, float* path_block, double g, int n_sweeps, int nt, double D)
{
	int i = blockIdx.x;
	float shift, dS, accept_prob;
	for(int sweep=0; sweep < n_sweeps; sweep++)
	{
		for(int t=i*nt+1; t < (i*nt+nt); t++)
		{
			shift = D*(2.0*curand_uniform(&state[i]) - 1);
			dS = shift*(shift + 2*path_block[t] - g*(path_block[t-1] + path_block[t+1]));
			accept_prob = fmin(1.0, expf(-1.0*dS));
			if(curand_uniform(&state[i]) <= accept_prob)
			{
				path_block[t] += shift;
			}
		}
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

	
	double eps = 0.01;
    double omega_eps = 0.25;
    double omega = omega_eps / eps;
    double kappa = 0.25*pow(eps, 2)*pow(omega, 2);
    double g = (1 - kappa) / (1 + kappa);

	int nt = 1000;
	int n_paths_block = 1024;
	int n_blocks = 10;
	int n_paths = n_blocks*n_paths_block;
	int n_sweeps = 100;
	double D = 1.5;
	
	curandState *d_state;
	cudaMalloc(&d_state, n_paths_block);

	size_t path_block_size = nt*n_paths_block*sizeof(float);
	float* path_block = (float*)malloc(path_block_size);
	for(int v=0; v<path_block_size; v++)
	{
		path_block[v]= 2*(((double)rand()/(double)RAND_MAX)) - 1.0;
	}
	

	float* cuda_path_block;
	cudaMalloc(&cuda_path_block, path_block_size);

	cudaMemcpy(cuda_path_block, path_block, path_block_size, cudaMemcpyHostToDevice);

	printf("Going PARALLEL!\n");
	init_rand<<<n_paths_block, 1>>>(d_state);
	mh_update<<<n_paths_block, 1>>>(d_state, cuda_path_block, g, n_sweeps, nt, D);
	cudaMemcpy(path_block, cuda_path_block, path_block_size, cudaMemcpyDeviceToHost);
	printf("Copied \n");
	write_paths_to_csv("cuda_paths_out.csv", path_block, n_paths_block, nt, "w");
	printf("Done first block.  Starting the rest\n");
	for(int blknum=1; blknum < n_blocks; blknum++)
	{
		mh_update<<<n_paths_block, 1>>>(d_state, cuda_path_block, g, n_sweeps, nt, D);
		cudaMemcpy(path_block, cuda_path_block, path_block_size, cudaMemcpyDeviceToHost);
		write_paths_to_csv("cuda_paths_out.csv", path_block, n_paths_block, nt, "a");
		printf("Done with block: %d\n", blknum);
	}

	cudaFree(d_state);
	cudaFree(cuda_path_block);
	free(path_block);
	return 0;
}

