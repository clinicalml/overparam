mkdir experiments_demo
python model_generation/generate_ground_truth.py experiments_demo
mkdir experiments_demo/samples
python sampling/generate_samples.py experiments_demo 0 1000 5
mkdir experiments_demo/runs
cd training
python run_train.py ../experiments_demo/samples/samples.txt ../experiments_demo/runs 48 2
cd ..
python visualization/evaluate_results.py experiments_demo/true_params/matrix.txt experiments_demo/runs model_epoch100.dat 2
