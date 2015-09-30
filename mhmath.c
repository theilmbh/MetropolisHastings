#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mh.h"

double rand_in_range(double min, double max)
{
    
    return min+(((double)rand() / (double)RAND_MAX )* (max-min));
    
}

double* generate_random_initial_path(int nt, double min, double max)
{

    int i;
    double *path = malloc(nt*sizeof(double));
    
    printf("Generating a random initial path of length %d\n", nt);
    for(i=0; i<nt; i++)
    {
        path[i] = rand_in_range(min, max);
    }
    return path;
    
}