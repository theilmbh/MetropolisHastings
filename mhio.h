/* mhio.h
 * Header file for input/output routines
 * Brad Theilman September 2015
 */
 
 #ifndef MHIO_H
 #define MHIO_H
 
 int write_to_csv(char* fname, double* data, int n_data);
 int print_path(double* path, int nt);
 void display_progress(int completed, int total, int cmp_mod);
 
 #endif