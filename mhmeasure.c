#include <stdlib.h>
#include "mh.h"

double* compute_correlation(double **paths, int n_paths, int te_0, int tau_eps_max)
{

    int i, te;
    double sum;
    double *corr = calloc(tau_eps_max, sizeof(double));
    for(te = 1; te<tau_eps_max; te++)
    {
        sum = 0;
        for(i=0; i<n_paths; i++)
        {
            sum += paths[i][te] * paths[i][1];
        }
        corr[te - 1] = sum / (double)n_paths;
    }
    
    return corr;
}
