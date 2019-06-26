#include "sampling.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

// Samples from a noisy-OR ground truth model.

int main(int argc, char* argv[])
{
    if(argc < 4)
    {
        cout << "usage: ./sample model_path seed n" << '\n';
        exit(1);
    }

    char * network_name = argv[1];
    int seed = atof(argv[2]);
    int n = atoi(argv[3]);

    cout << "sampling from " << network_name << " network taking " << n << " samples with seed " << seed << '\n';

    Sampler s = Sampler(network_name, seed, n);
    s.sample(true);
}
