## Neural PCFG Experiments

We use the same dataset/preprocessing from the [compound PCFG repo](https://github.com/harvardnlp/compoun-pcfg), which is available here [here](https://drive.google.com/file/d/1m4ssitfkWcDSxAE6UYidrP6TlUctSG2D/view?usp=sharing).

After downloading, run the preprocessing script
```
python preprocess.py --trainfile data/ptb-train.txt --valfile data/ptb-valid.txt 
--testfile data/ptb-test.txt --outputfile ptb --vocabsize 10000 --lowercase 1 --replace_num 1
```
Running this will save the following files in the current folder: `ptb-train.pkl`, `ptb-val.pkl`, `ptb-test.pkl`, `ptb.dict`. Here `ptb.dict` is the word-idx mapping, and you can change the output folder/name by changing the argument to `--outputfile`.

Then train the neural PCFG on PTB with
```
python train_ptb.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl 
--save_path ptb-neural-pcfg.pt --gpu 0
```
where `--save_path` is where you want to save the model, and `--gpu 0` is for using the first GPU in the cluster (the mapping from PyTorch GPU index to your cluster's GPU index may vary).

Now we use the above model as our data-generating distribution to create a semi-synthetic dataset.

```
python sample.py --train_samples 50000 --dev_samples 1000 --test_samples 1000 --model_file ptb-neural-pcfg.pt --gpu 0 --output_file pcfg-data.pkl
```
Note that we overgenerate the number of training samples, and simply train on fewer samples during training if need be. 

Next we train the neural PCFG on generated data
```
python train_sample.py --data_file pcfg-data.pkl --t_states 10 --nt_states 10 --save_path sample-neural-pcfg.pt --gpu 0 --train_samples 5000
```
Here `--t_states` controls the number of preterminals, and `--nt_states` specifies the number of nonterminals. These are tweaked to train models with varying degrees of overparameterization.

Finally, for evaluation to calculate NLL/F1 on test set, run
```
python eval.py --data_file pcfg-data.pkl --model_file sample-neural-pcfg.pt --gpu 0 
```
