mkdir experiments_demo
python model_generation/generate_synth_images.py experiments_demo # generates model parameters
mkdir experiments_demo/samples # directory in which samples will be stored
cd sampling
rm *.o
rm sample
make
cd ..
./sampling/sample experiments_demo 0 1000 # generates dataset (i.e. generates samples from model parameters)
mkdir experiments_demo/runs # directory in which experiment results will be stored
cd training
python run_train_recognition_network.py ../experiments_demo/samples/raw_samples_n1000_s0 ../experiments_demo/runs 16 2 # trains the algorithm on the dataset, with 16 latent variables, and 2 runs with different random seeds
cd ..
python visualization/evaluate_results.py experiments_demo/true_params experiments_demo/runs 100 2 # evaluates the results (e.g. how many latent variables match ground truth latent variables)
python visualization/visualize_images.py experiments_demo/runs/R1/model_epoch100.dat # visualizes the resulting model in the first run
python visualization/visualize_images.py experiments_demo/runs/R2/model_epoch100.dat # visualizes the resulting model in the second run
