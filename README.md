# Using Reinforcement Learning for Multi-Objective Cluster-Level Optimization of Non-Pharmaceutical Interventions for Infectious Disease
Accepted at **ML4H-2023** 
> Author: **Xueqiao Peng**, Jiaqi Xu, Xi Chen, Dinh Song An Nguyen, Andrew Perraul\
> Email: peng.969@osu.edu

# Install
+ Dependency: `Python 3.6`, `Pytorch 1.10.2`,`Spinning Up 0.2.0`, `Gym 0.15.7`
+ Install Conda to set up an virtual environment [Toturial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart).
+ Install Pytorch [Toturial](https://pytorch.org/get-started/locally/).
+ Install Spiningup [Toturial](https://spinningup.openai.com/en/latest/user/installation.html/).

# Run
+ Supervised learning model:
  - Collect data from simulator ``python3 data_generator_test.py`` 
  - Then train supervised learning model ``python3 Supervised_Learning_test.py``
+ Reinforcement learning model:
  - Using the trained SL model, execute ``python3 -m spinup.run ppo --exp_name [example name] --env [environment name] --epochs [epochs numner] --seed [seed]``, and collect the input and output
  - Using the new dataset retrain the SL model.
  - Using the retrained SL model, execute ``python3 -m spinup.run ppo --exp_name [example name] --env [environment name] --epochs [epochs numner] --seed [seed]`` again.
  - Evaluation: ``python3 -m spinup.run test_policy [path/to/output_directory]``
+ Comparison Policies:
  - Change the environment name. Noted that no retrained SL model for other policies.
+ Interpratable Policy:
  - Get features from trained policies and execute ``python3 decisiontree.py``
  - Evaluation: ``python3 DT_eval.py --dpath [DT_directory] --fpath [path/to/output_directory]``
  
