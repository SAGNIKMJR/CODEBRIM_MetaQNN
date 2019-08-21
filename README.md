# CODEBRIM_MetaQNN
This is my open-source implementation of the Q-learning based MetaQNN neural architecture search algorithm to find suitable convolutional neural network architectures for our challenging multi-class multi-target *CODEBRIM* bridge-defect recognition dataset. The entire code has been written in *Python 3.5.2* (although the code should work in principle with other *Python 3* versions as well) and *PyTorch 1.0.0*. The official code for our paper "Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset": [IEEE open access](http://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html) or [https://arxiv.org/abs/1904.08486](https://arxiv.org/abs/1904.08486) is at [this link](https://github.com/MrtnMndt/meta-learning-CODEBRIM)
## Installing code dependencies
`pip3 install -r requirements.txt`

## Running a search
A search can be conducted by simply executing

`python3 main.py -t 1 --dataset-path PATH_TO_CODEBRIM_DATASET`

This will launch the MetaQNN search with 200 architectures and the exact hyperparameters as specified in the paper. Before running the search, the *CODEBRIM* dataset needs to be downloaded from https://zenodo.org/record/2620293#.XPLE_nUzY8o , unzipped and the path to this directory should be subsituted for *PATH_TO_CODEBRIM_DATASET*. All necessary search and training hyperparameters are exposed in the command line parser that can be found in *lib/cmdparser.py*.

## Resuming an incomplete search
An incomplete search can be resumed by executing

`python3 main.py -t 1 --dataset-path PATH_TO_CODEBRIM_DATASET --q-values-csv-path PATH_TO_LAST_Q_VALUES --replay-buffer-csv-path PATH_TO_LAST_REPLAY_BUFFER`

where the additional arguments, *PATH_TO_LAST_Q_VALUES* and *PATH_TO_LAST_REPLAY_BUFFER* are paths to the most recently saved *Q-values* and *replay buffer* csv files during the incomplete search.

## Retraining an architecture from the search
To retrain and revaluate an architecture from a previous search, one may execute

 `python3 main.py -t 2 --dataset-path PATH_TO_CODEBRIM_DATASET --replay-buffer-csv-path PATH_TO_LAST_REPLAY_BUFFER --fixed-net-index-no INDEX_NO_OF_ARC`
 
 Here, the additional parameter, *INDEX_NO_OF_ARC* denotes the index of the architecture configuration which needs to be retrained, as per the indexing in the replay buffer.
