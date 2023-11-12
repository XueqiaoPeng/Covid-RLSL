#!/bin/bash
#SBATCH --account=PAS2138
#SBATCH --job-name=RLSL
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --output=RLSL.out.%j

# V0-RLSL
# python3 -m spinup.run ppo1 --exp_name test --env CovidWorld-v0 --epochs 30 --seed 5000;
# cd /users/PCON0023/xueqiao/miniconda3/envs/spinningup/lib/python3.6/site-packages/gym/envs/CovidRL/
# python3 Supervised_Learning_test.py --train ./new_data/feature0.csv --test ./new_data/infection0.csv --model ./model/model_new0.pth 
# python3 -m spinup.run ppo --exp_name RL4 --env CovidWorld-v0 --epochs 200 --seed 5000;
# for i in $(seq 0 29)  
# do   
#     python3 -m spinup.run test_policy /users/PCON0023/xueqiao/spinningup/data/RL4/RL4_s5000;
# done 


#V1-benchmark
# python3 -m spinup.run ppo --exp_name CDC070 --env CovidWorld-v1 --epochs 25 --seed 0;
# for i in $(seq 0 29)  
# do   
#     python3 -m spinup.run test_policy /users/PCON0023/xueqiao/spinningup/data/CDC070/CDC070_s0;
# done 


#V2-DailyTest
# python3 -m spinup.run ppo --exp_name TE005001 --env CovidWorld-v2 --epochs 150 --seed 4997;
# for i in $(seq 0 29)  
# do   
#     python3 -m spinup.run test_policy /users/PCON0023/xueqiao/spinningup/data/TE005001/TE005001_s4997;
# done 


# V3-pureRL
# python3 -m spinup.run ppo --exp_name pureRL005001 --env CovidWorld-v3 --epochs 200 --seed 4997;
# for i in $(seq 0 29)  
# do   
#     python3 -m spinup.run test_policy /users/PCON0023/xueqiao/spinningup/data/pureRL005001/pureRL005001_s4997;
# done 


#V4-NoTest
# python3 -m spinup.run ppo --exp_name notest02 --env CovidWorld-v4 --epochs 200 --seed 4997;
# for i in $(seq 0 29)  
# do   
#     python3 -m spinup.run test_policy /users/PCON0023/xueqiao/spinningup/data/notest02/notest02_s4997;
# done 

