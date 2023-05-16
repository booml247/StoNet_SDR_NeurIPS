## CONTENTS OF THIS FILE
* Introduction
* Requirements
* Usage
* License
## INTRODUCTION
Sufficient dimension reduction is a powerful tool to extract core information hidden in the high-dimensional data and has potentially many important applications in machine learning tasks. However, the existing state-of-the-art nonlinear sufficient dimension reduction  methods often involve eigen-decomposition of the Gram matrix calculated in the reproducing kernel Hilbert space, making them lack the scalability necessary for dealing with large-scale data. We propose a new type stochastic neural network under a rigorous probabilistic framework and show that it can be used for sufficient dimension reduction for high-dimensional data. The proposed stochastic neural network can be trained using an adaptive stochastic gradient Markov chain Monte Carlo algorithm, which is scalable by the use of mini-batch data in iterations. For more details, please refer to the paper.

## REQUIREMENTS
* pytorch: Refer [Pytorch](http://pytorch.org/) for installation instructions.
* torchvision: Refer [Torchvision](https://pypi.org/project/torchvision/) for install instructions.

## USAGE
### Data Preparation
For real data example, data needs to be downloaded and stored in the "./data/" folder. Please see process_data.py for more details.

### Binary Classification Example
This is an example of how to do sufficient dimension reduction (SDR) using stonet and then train a logistic regression model on the projected predictors for the binary classification task on the "breast cancer" dataset.
```
python run_stonet.py --data_name 'breast cancer' --num_epoch 100 --batch_size 32 --MH_step 25 --sigma_list [1e-5, 1e-8] --alpha 0.1 --proposal_lr [5e-9] --lr 0.001 --regression_flag False --net_architecture [4, 1] --model_type "Logistic"
```
For the other datasets such as "flare solar", "german" etc., the input of `data_name` and other hyperparameters should be changed accordingly.


### Multi-label Classification Example
This is an example of how to do SDR using stonet on MNIST dataset.
```
python run_stonet.py --data_name 'MNIST' --num_epoch 20 --batch_size 128 --MH_step 25 --sigma_list [1e-3, 1e-6] --alpha 0.1 --proposal_lr [5e-7] --lr 0.001 --regression_flag False --net_architecture [98, 10]
```

To do multi-label classification on the dimension reduced data using DNN, run the following script in command line.
```
python run_DNN.py --data_name 'MNIST' --num_epoch 100 --batch_size 128 --lr 0.001 --regression_flag False --net_architecture [50, 10]
```
### Regression Example
This is an example of how to do SDR using stonet on relative location of CT slices on axial axis dataset.
```
python run_stonet.py --data_name 'slice_localization' --num_epoch 100 --batch_size 800 --MH_step 1 --sigma_list [1e-7, 1e-8, 1e-8] --alpha 0 --proposal_lr [5e-9, 5e-10] --lr 0.01 --regression_flag True --net_architecture [400, 192, 1]
```
To do regression on the dimension reduced data using DNN, run the following script in command line.
```
python run_DNN.py --data_name 'slice_localization' --num_epoch 100 --batch_size 800 --lr 0.01 --regression_flag True --net_architecture [100, 1]
```
## LICENSE
Distributed under the MIT License. See  `LICENSE`  for more information.
