# ON-POLICY

This repo is only used for test. 

If there are any problems about this repo, feel free to contact me: yc19@mails.tsinghua.edu.cn.

## support algorithms

| Algorithms | recurrent-verison | mlp-version | cnn-version | mixed-base version | independent version |
| :--------: | :---------------: | :---------: | :---------: |:---------------: |:---------------: |
| MAPPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark:        |:heavy_check_mark:        |


## support environments:
**Pay Attention:** we sometimes hack the environment code to fit our task and setting. 
- [MPE](https://github.com/openai/multiagent-particle-envs)
- [Mini-GridWorld](https://github.com/maximecb/gym-minigrid)

## 1. Install

### 1.1 instructions

   test on CUDA == 10.1, of course other versions can also works.

``` Bash
   conda create -n marl
   conda activate marl
   pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
   cd onpolicy-test
   pip install -e . 
```

### 1.2 hyperparameters

* config.py: contains all hyper-parameters

* default: use GPU, chunk-version recurrent policy and shared policy

* other important hyperparameters:
  - use_centralized_V: Centralized training (MA) or Centralized training (I)
  - use_single_network: share base or not
  - use_recurrent_policy: rnn or mlp
  - use_eval: turn on evaluation while training, if True, u need to set "n_eval_rollout_threads"
  - wandb_name: For example, if your wandb link is https://wandb.ai/mapping, then you need to change wandb_name to "mapping". 
  - user_name: only control the program name shown in "nvidia-smi".


## 2. MPE
MPE is kept here as the simple example to learn on-policy repo.

### 2.1 Install MPE

``` Bash
   # install this package first
   pip install seabon
```

3 Cooperative scenarios in MPE:

* simple_spread: set num_agents=3
* simple_speaker_listener: set num_agents=2, and use --share_policy
* simple_reference: set num_agents=2

### 2.2 Train MPE   

``` Bash
   conda activate marl
   cd scripts
   chmod +x train_mpe.sh
   ./train_mpe.sh
```

## 3. gym-GridWorld

Here is the true task. 

What we have done:
- support multi-agents
- the gridworld runner, fyi

What needs to be done:
- implemment the multi-agent mapping scenario.
- make it work!

## 4. Docs：

```
pip install sphinx sphinxcontrib-apidoc sphinx_rtd_theme recommonmark

sphinx-quickstart
make html
```
