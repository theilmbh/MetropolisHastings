#include "mh.h"


/* Action definitions */

double S_euclidean(double *path, double g, int nt)
{

    /* Returns the value of the action for a path */
    int i, j;
    double A = 0;
    double B = 0;
    
    for(j = 0; j < nt-1; j++)
    {
        A += path[j+1]*path[j+1];
        B += g*path[j]*path[j+1];
    } 
    
    return A+B;

}

double delta_S(double* path, double g, double delta, int path_index)
{

    /* Returns value of change in action due to updating value at path_index by delta */
    return delta*(delta + 2*path[path_index] - g*(path[path_index-1] + path[path_index+1]));

}