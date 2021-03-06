example.ipynb shows how to generate a dataset, how to run the training algorithm on it, and how to evaluate/visualize the results.

datasets/ contains the actual data sets used in the experiments. For each data set, true_params/ contains the parameters of the ground truth model and samples/ contains the samples generated from the ground truth model.

experiments_config/ contains, for each experiment reported in the paper, a config.txt file. Each line of a config.txt file represents the hyperparameters and random seed used in one run of that experiment (corresponding to the config object in the training file). Note that all runs of an experiment use the same hyperparameters, and only the random seed differs (i.e. the only difference between lines in a config.txt file is in the "seed" field).
