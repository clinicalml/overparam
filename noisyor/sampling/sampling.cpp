#include <iostream>
#include <fstream>
#include <math.h>
#include "sampling.h"
#include <stdlib.h>
#include "mtrand.h"

using namespace std;


Sampler::Sampler(char* dirname, int sample_seed, int numSamples){
    verbose=false;

    ifstream f;
    char fname[100];
    seed = sample_seed;

    network_name = dirname;
    num_samples = numSamples;

    sprintf(fname, "%s/true_params/priors.txt", dirname);
    f.open(fname);
    f >> P;
    priors = new float[P];
    for (int i = 0; i < P; i++)
    {
        f >> priors[i];
    }
    f.close();
    printf("read %d priors\n", P);

    sprintf(fname, "%s/true_params/noise.txt", dirname);
    f.open(fname);
    f >> C;
    noise = new float[C];
    for (int i = 0; i < C; i++)
    {
        f >> noise[i];
    }
    f.close();
    printf("read %d noises\n", C);

    sprintf(fname, "%s/true_params/weights.txt", dirname);
    f.open(fname);
    edges = new float*[P];
    for (int i = 0; i < P; i++){
        edges[i] = new float[C];
        for (int j = 0; j < C; j++)
        {
            f >> edges[i][j];
        }
    }

    f.close();
    printf("read weights\n");
}

void Sampler::sample(bool printSamples)
{
    MTRand_closed rnd;
	rnd.seed((unsigned long int)seed);
    char* dirname = network_name;
    int obs[C];
    float r;

    ofstream out;
    char fname[256];
    sprintf(fname, "%s/progress_log_n%d_s%d", network_name, num_samples, seed);
    out.open(fname);

    for(int sample_counter = 0; sample_counter < num_samples; sample_counter++)
    {
        if((sample_counter  % 1000) ==0)
        {
            printf("sample %d\n", sample_counter);
        }

        for (int c = 0; c < C; c++){
            //initialize with noise
            r = rnd();
            if(r > noise[c]) {
                obs[c] = 1;
            }
            else {
                obs[c] = 0;
            }
        }

        for(int p = 0; p < P; p++)
        {
            r = rnd();
            if(r < priors[p])
            {
                for(int c = 0; c < C; c++)
                {
                    if(edges[p][c] < 1 && rnd() > edges[p][c]) {
                        obs[c] = 1;
                    }
                }
            }
        }

        //obs is filled with 0's and 1s now.
        if(printSamples)
        {
            ofstream out;
            char fname[256];
            sprintf(fname, "%s/samples/raw_samples_n%d_s%d", network_name, num_samples, seed);
            if (sample_counter) out.open(fname, ofstream::app);
            else out.open(fname);
            for (int c = 0; c < C-1; c++){
                out << obs[c] << " ";
            }
            out << obs[C-1];
            out << '\n';
            out.close();
        }
    }
    out.close();
}
