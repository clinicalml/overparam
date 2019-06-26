using namespace std;

class Sampler {

    bool verbose;

    int P;
    int C;
    int N;
    int seed;
    float* noise;
    float* priors;
    float** edges;

    char* network_name;
    char* raw_samples_name;
    int num_samples;

public:
    Sampler(char* dirname, int seed, int num_samples);
    void sample(bool printSamples);
};
