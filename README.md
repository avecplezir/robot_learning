[![tests](https://github.com/milarobotlearningcourse/robot_learning/actions/workflows/testing.yaml/badge.svg)](https://github.com/milarobotlearningcourse/robot_learning/actions/workflows/testing.yaml)


# Setup

You can run this code on your own machine or on Google Colab (Colab is not completely supported). 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) from homework 1 for instructions. There are two new package requirements (`opencv-python` and `gym[atari]`) beyond what was used in the previous assignments; make sure to install these with `pip install -r requirements.txt` if you are running the assignment locally.

2. **Docker:** You can also run this code in side of a docker image. You will need to build the docker image using the provided docker file

3. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:

### Anaconda 

If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux).
Then create a anaconda environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).

```
conda create -n roble python=3.8
conda activate roble
pip install -r requirements.txt
```

If having issues with Pytorch and GPU, make sure to install the compatible version of Pytorch for your CUDA version [here](https://pytorch.org/get-started/locally/)



## Examples:

Please use following commands to produce figure in the report:

Base BC Ant agents:

```
python run_hw1_bc.py logging.random_seed=1
python run_hw1_bc.py logging.random_seed=2
python run_hw1_bc.py logging.random_seed=3
```

Base BC Humanoid agents:
```
python run_hw1_bc.py logging.random_seed=1 env.env_name=Humanoid-v2 env.expert_policy_file='../../../hw1/roble/policies/experts/Humanoid.pkl' env.expert_data='../../../hw1/roble/expert_data/expert_data_Humanoid-v2.pkl'
```

Varying batch size for BC Ant agents:
```
python run_hw1_bc.py logging.random_seed=2 alg.batch_size=200
python run_hw1_bc.py logging.random_seed=2 alg.batch_size=500
python run_hw1_bc.py logging.random_seed=2 alg.batch_size=1000
python run_hw1_bc.py logging.random_seed=2 alg.batch_size=2000
```
IDM for Ant agents:

```
python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false alg.train_idm=true logging.random_seed=2
```

BC and IDM for HalfCheetah agents:

```
python run_hw1_bc.py logging.random_seed=1 env.env_name=HalfCheetah-v2 env.expert_policy_file='../../../hw1/roble/policies/experts/HalfCheetah.pkl' env.expert_data='../../../hw1/roble/expert_data/expert_data_HalfCheetah-v2.pkl'
python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false alg.train_idm=true logging.random_seed=1 env.env_name=HalfCheetah-v2 env.expert_policy_file='../../../hw1/roble/policies/experts/HalfCheetah.pkl' env.expert_data='../../../hw1/roble/expert_data/expert_data_HalfCheetah-v2.pkl' env.expert_unlabelled_data=../../../hw1/roble/expert_data/unlabelled/unlabelled_data_HalfCheetah-v2.pkl
```

Dagger Ant

```
python run_hw1_bc.py alg.n_iter=5 alg.do_dagger=true alg.train_idm=false logging.random_seed=2
```

Dagger Humanoid

```
python run_hw1_bc.py alg.n_iter=20 alg.do_dagger=true alg.train_idm=false logging.random_seed=1 alg.batch_size=1000 env.env_name=Humanoid-v2 env.expert_policy_file='../../../hw1/roble/policies/experts/Humanoid.pkl' env.expert_data='../../../hw1/roble/expert_data/expert_data_Humanoid-v2.pkl'
```

salloc -c 4 --gres=gpu:1 --mem=15G
jupyter notebook --no-browser --ip=* --port=8081
ssh -L 8081:cn-a011:8081 -fN mila


Assignments for [UdeM roble: Robot Learning Course](https://fracturedplane.com/teaching-new-course-in-robot-learning.html). Based on [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

