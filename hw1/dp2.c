#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4)
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
    return R;
}

int main(int argc, char *argv[] )  {
    
    if( argc == 3 ) {
        long n = atoi(argv[1]);
        long iter = atoi(argv[2]);
        long i;
        float R = 0;
        struct timespec start, end;
        float *pA = (float *) malloc(n*sizeof(float));
        float *pB = (float *) malloc(n*sizeof(float));
        memset(pA, 0x3f, n*sizeof(float));
        memset(pB, 0x3f, n*sizeof(float));
        float total_time = 0;

        for(i = 0; i<iter; i++){
            clock_gettime(CLOCK_MONOTONIC,&start);
            R = dpunroll(n,pA,pB);
            clock_gettime(CLOCK_MONOTONIC,&end);
            
            double time_usec = (((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
            if (i >= iter/2) {
                total_time += time_usec;
            }
        }

        double avg_time = total_time / (iter/2);
        double bw = 2 * n * sizeof(float) / (avg_time / 1e6)/ (1024 * 1024 * 1024);
        double flops = 2 * n / ((avg_time / 1e6) * 1e9);
        double ai = flops/bw;
        printf("%f",R);

        printf("\n\ndp2.c\n");
        printf("Total_time: %0.6f\n",total_time);
        printf("N: %ld <T in sec>: %0.6f sec <T in usec>: %0.6f usec B: %0.6f GB/sec F: %0.6f GFLOP/sec ai: %0.6f FLOP/byte\n",n,avg_time / 1e6 ,avg_time,bw,flops,ai);

        free(pA);
        free(pB);
    }
    else {
        printf("two argument expected.\n");
    }
}