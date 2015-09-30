/* lagrangian.h
 * Headerfile for lagrangian declarations
 * Brad Theilman September 2015
 */
 
 #ifndef LAGRANGIAN_H
 #define LAGRANGIAN_H
 
 double S_euclidean(double *path, double g, int nt);
 double delta_S(double* path, double g, double delta, int path_index);
 
 #endif