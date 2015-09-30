#include <stdio.h>
#include <stdlib.h>

#include "mh.h"


void display_progress(int completed, int total, int cmp_mod)
{

    /* clear line */
    /*printf("\33[2K\r");*/
    if(completed % cmp_mod == 0)
    {
        double pct_complete = 100* ((double)completed / (double)total);
        printf("%d of %d (%.2f%%) Completed.\n", completed, total, pct_complete);
    }

}

int print_path(double* path, int nt)
{
    /* print path */
    printf("---- CURRENT PATH VALUES ---- \n");
    for(int i = 0; i<nt; i++)
    {
        printf("%f\n", path[i]);
    }
    printf("\n");
    return 0;
}

int write_to_csv(char* fname, double* data, int n_data)
{
	
	FILE *outfile;
	outfile = fopen(fname, "w");
	
	int i;
	
	if(outfile != NULL)
	{
		for(i = 0; i<n_data; i++)
		{
			fprintf(outfile, "%f\n", data[i]);
		}
	}
	fclose(outfile);
	return 0;
}
