# *n*FBST

The code of AAAI'24 paper *n*FBST. [Full Bayesian Significance Testing for Neural Networks | Abstract (arxiv.org)](https://arxiv.org/abs/2401.13335)

Significance testing aims to determine whether a proposition about the population distribution is the truth or not given observations. However, traditional significance testing often needs to derive the distribution of the testing statistic, failing to deal with complex nonlinear relationships. In this paper, we propose to conduct Full Bayesian Significance Testing for neural networks, called *n*FBST, to overcome the limitation in relationship characterization of traditional approaches. A Bayesian neural network is utilized to fit the nonlinear and multi-dimensional relationships with small errors and avoid hard theoretical derivation by computing the evidence value. Besides, *n*FBST can test not only global significance but also local and instance-wise significance, which previous testing methods don't focus on. Moreover, *n*FBST is a general framework that can be extended based on the measures selected, such as Grad-*n*FBST, LRP-*n*FBST, DeepLIFT-*n*FBST, LIME-*n*FBST.  A range of experiments on both simulated and real data are conducted to show the advantages of our method. 

If our work has been of assistance to you, please feel free to cite our paper. Thank you.

```
@inproceedings{liu2024full,
  title={Full Bayesian Significance Testing for Neural Networks},
  author={Liu, Zehua and Li, Zimeng and Wang, Jingyuan and He, Yue},
  booktitle={Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

# Pipeline

## Before reading

* Note you need to use the command `--log True` cautiously.

  It means the operation will change the following pipeline. Specifically, the log and related files are only saved in the log directory. However, if setting `--log True`, the results will be saved in the results directory. This command is usually used for the first time.

* Toy example in the paper corresponds to simulation_v3 in the code.

* Dataset 1 and Dataset 2 in the paper correspond to simulation_v4 and simulation_v12 in the code respectively.

## Pipeline for Toy Example

### 1 Data Generation Process

Generate different types of simulation datasets.

```shell
python generate_simulation_data.py --log True --data simulation_v3
```

#### 1.2 Binary Label Generation Process

Generate binary label for instance-wise significance and global significance.

Here, `eps` only affects instance-wise significance but not global significance.

```shell
python generate_simulation_data_label.py --log True --data simulation_v3 --eps 0.001
```

### 2 Train Model

#### 2.0 Grid Search (Optional)

Before training, one can run grid search for the best hyper-parameters.

The hyper-parameters you would like to adjust are needed to set in the code. Then, save the best ones in `models/model_config.py` file.

```shell
python grid_search.py --log True --data simulation_v3 --model_type gaussian
python grid_search.py --log True --data simulation_v3 --model_type nn
```

Training.

```shell
python train_model.py --log True --data simulation_v3 --model_type gaussian
python train_model.py --log True --data simulation_v3 --model_type nn
```

### 3 Distribution of Testing Statistics Generation Process

```shell
python get_statistic_distri.py --log True --data simulation_v3 --model_type gaussian --model_name gaussian_1/gaussian_2/gaussian_3 --interpret_method gradient/DeepLIFT/LRP/LIME
python get_statistic_distri.py --log True --data simulation_v3 --model_type nn --sample_T 3 --model_name nn_1 --interpret_method gradient/DeepLIFT/LRP/LIME
```

#### 3.1 Distribution of GradientXInput Generation Process (Optional)

The results are similar to LRP. The paper (*Kindermans, Investigating the influence of noise and distractors on the interpretation of neural networks.*) shows that the LRP rules for ReLU networks are equivalent within a scaling factor to gradient × input in some conditions.

```shell
python get_statistic_distri.py --log True --data simulation_v3 --model_type gaussian --model_name gaussian_1/gaussian_2/gaussian_3 --interpret_method gradientXinput
python get_statistic_distri.py --log True --data simulation_v3 --model_type nn --sample_T 3 --model_name nn_1 --interpret_method gradientXinput
```

#### 3.2 Ensemble Different BNNs

In practice, we adopt diagonal Gaussian distribution as our variational family of model parameters and ensemble three Bayesian neural networks to spread the range of model parameters further.

```shell
python ensemble_statistic_distri.py --log True --data simulation_v3 --model_type gaussian --model_name gaussian_e --interpret_method gradient/DeepLIFT/LRP/LIME
```

### 4 Bayesian Evidence Generation Process

```shell
python get_bayes_factors.py --log True --data simulation_v3 --model_type gaussian --model_name gaussian_e --interpret_method gradient/DeepLIFT/LRP/LIME --algorithm p_s
python get_bayes_factors.py --log True --data simulation_v3 --model_type nn --model_name nn_1 --interpret_method gradient/DeepLIFT/LRP/LIME --algorithm mean
```

#### 4.2 Merge Evidence (Optional)

This is only for accelerating. Chunks run Step 4 in parallel and then merge.

Note the timestamps to be merged need to be set in the code.

```shell
python ensemble_bayes_factors.py --log True --data simulation_v3 --model_type gaussian --model_name gaussian_e --interpret_method gradient/DeepLIFT/LRP/LIME --algorithm p_s
```

### 5 Run Baselines

#### 5.1 Bootstrap

```shell
python run_bootstrap.py --data simulation_v3 --n_samples_per_bootstrap 1000
```

#### 5.2 *t*-test

```shell
python run_t_test.py --data simulation_v3
```

#### 5.3 likelihood-ratio test

```shell
python run_likelihood_ratio_test.py --data simulation_v3
```

### 6 Analyze

#### 6.1 Sort All Bayesian Evidence

```shell
python local_2_global.py --data simulation_v3 --eps 0.001 --interpret_method gradient/DeepLIFT/LRP/LIME
```

#### 6.2 Plot Scatters and Histograms

```shell
python plot_simulation_v3_scatter.py --interpret_method gradient/DeepLIFT/LRP/LIME
```

## Pipeline for Simulation Experiments

From Step 1 to Step 5 is the same as `Pipeline for Toy Example`, just pay attention to change `--data simulation_v4/simulation_v12`.

### 6 Analyze

From Step 6, 6.1 is also applicable, and we have some other analyses.

#### 6.1 Sort All Bayesian Evidence

```shell
python local_2_global.py --data simulation_v4/simulation_v12 --eps 0.001 --interpret_method gradient/DeepLIFT/LRP/LIME
```

#### 6.2 Compare with Feature Importance Analysis

```shell
python analyse_bayes_factors.py --log True --data simulation_v4/simulation_v12 --model_type gaussian --model_name gaussian_e --eps 0.001/0.01/0.02/0.03/0.04/0.05 --interpret_method gradient/DeepLIFT/LRP/LIME --algorithm p_s

python analyse_bayes_factors.py --log True --data simulation_v4/simulation_v12 --model_type nn --model_name nn_1 --eps 0.001/0.01/0.02/0.03/0.04/0.05 --interpret_method gradient/DeepLIFT/LRP/LIME --algorithm mean
```

#### 6.3 Plot ROC and Joint Together

Note this maybe needs to fix `interpret_method, algorithms, model_names` in the code.

```shell
python joint_different_curves.py --data simulation_v4/simulation_v12 --eps 0.001/0.01/0.02/0.03/0.04/0.05 --control all/gradient/DeepLIFT/LRP/LIME
```

#### 6.4 Plot Average AUC under Different Eps

```shell
python plot_avg_auc.py
```

## Pipeline for Energy Efficient

From Step 1 to Step 4 is the same as `Pipeline for Toy Example`, just pay attention to change `--data energy_16`.

### 6 Analyze

From Step 6, 6.1 is also applicable, and we have some other analyses.

#### 6.1 Sort All Bayesian Evidence

```shell
python local_2_global.py --data energy_16 --eps 0.001 --interpret_method gradient/DeepLIFT/LRP/LIME
```

#### 6.2 Plot $x_8$ Distribution

```shell
python analyse_energy.py --data energy_16 --interpret_method gradient/DeepLIFT/LRP/LIME
```

## Pipeline for MNIST

### 1 Data Generation Process

For Step 1, there is a little difference from the above.

Select the corresponding serial number according to the classification label for the following comparison.

```shell
python generate_mnist_targets.py --log True
```

From Step 2 to Step 4 is the same as `Pipeline for Toy Example`, just pay attention to change `--data mnist`.

**There is another big difference, that is we need to set `--y_index 0/1/2/3/4/5/6/7/8/9`** because it is a classification task now. The setting is needed for Step 3, Step 4 except Step 2.

### 5 Saliency Map Comparison

Note this process may be slow, and we needn't run on the whole dataset. Please set the directory in the code.

```shell
python run_exp_mnist2.py
```

