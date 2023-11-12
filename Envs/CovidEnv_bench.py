# -*- coding: utf-8 -*-
# @Author: xueqiao
# @Date:   2023-02-09 12:50:33
# @Last Modified by:   xueqiao
# @Last Modified time: 2023-11-12 11:39:22
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import bernoulli
from .Supervised_Learning_test import NeuralNetwork
import torch
import torch.nn.functional as F


class CovidEnvbench(gym.Env):

    def __init__(self):
        self.size = np.random.randint(2,40,1)[0]
        self.days = 30  # Assume we observe the 2nd generation for 30 days
        # We assume weight_infect_no_quarantine = -1, and calculate weight_no_infect_quarantine from the ratio
        self.ratio = 0.70
        self.weights = self.ratio/(1 + self.ratio)
        self.ratio2 = 0.01
        self.p_high_transmissive = 0.109  # Probability that the index case is highly transmissive
        self.p_infected = 0.03  # Probability that a person get infected (given index case is not highly transmissive)
        self.p_symptomatic = 0.8  # Probability that a person is infected and showing symptom
        self.p_symptom_not_infected = 0.01  # Probability that a person showing symptom but not infected
        self.observed_day = 3  # The agent starts to get involved in day 3
        self.duration = 14  # The default quarantine duration is 14
        self.test = 0
        self.today_result = 0
        self.first_symptom = None
        self.simulated_state = None
        self.input_data = None
        self.prediction = None
        self.current_state = None
        self.case_test = None
        self.case_result = None
        C  = np.array([[34,22],[14,1869]])
        self.confusion_matrix = C / np.sum(C , axis=0)
        # Initialize the model
        self.model = NeuralNetwork().double()
        self.model.load_state_dict(torch.load('/users/PCON0023/xueqiao/miniconda3/envs/spinningup/lib/python3.6/site-packages/gym/envs/CovidRL/model/model_822.pth'))
        self.sum1 = 0
        self.sum2 = 0
        self.sum3 = 0
        self.case = None
        self.case_action  = None


        """
        Use a long vector to represent the observation. Only focus on one 2nd generation.
        We assume that person get exposed at day 0. 
        Index 0 to 2 represent the prediction of whether infected during three days before the observing day. 
        Index 3 to 5 represent the prediction of whether infected during three days after the observing day. 
        Index 6 to 8 represent whether that person shows symptoms in recent three days.
        Index 9 to 11 represents whether testing
        Index 12 to 14 represents the result of testing
        Index 15 to 17 represents how many tests ran in past three days
        Index 18 to 20 represents the cluster size
        """
        self.observation_space = spaces.Box(low=0, high=50, shape=(12,), dtype=np.float32)

        """
        0 for not quarantine ,1 for quarantine  
        
        """
        self.action_space = spaces.Discrete(2)

        self.seed()
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

   # We start to trace when the 1st generation case is tested positive.
    def reset(self, return_info=False):
        self.size = np.random.randint(2,40,1)[0]
        self.observed_day = 3
        self.duration = 14
        self.test = 0
        self.case = 0
        self.flag = 0
        self.simulated_state = self._simulation()
        # Initialize the current state
        self.current_state = np.zeros(12)
        # Find when first show symptom
        self.first_symptom = np.full(self.size, -1)
        self.case_test = np.zeros((self.size, self.days))
        self.case_result = np.zeros((self.size, self.days))
        self.case_action = np.zeros((self.size, self.days))
        self.sumtest = np.zeros(self.days)
        self.sumpositive = np.zeros(self.days)
        for i in range(self.size):
            for j in range(self.days):
                if self.simulated_state["Showing symptoms"][i][j] == 1:
                    self.first_symptom[i] = j
                    break
        self.prediction = np.zeros(self.days)
        # Build the input data
        self.input_data = np.zeros((9, self.days))
        if self.first_symptom[0] != -1:
            self.input_data[0][self.first_symptom[0]+1: self.days] = 1
        self.input_data[1][self.observed_day+1: self.days] = 1
        for day in range(0, self.days):
            symptom_num = 0.0
            for i in range(1, self.size):
                if self.first_symptom[i] <= day and self.first_symptom[i] != -1:
                    symptom_num += 1
            self.input_data[2][day] = symptom_num
        self.input_data[3] = self.size - 1
        self.input_data[4] = range(0, 30)
        self.input_data[5] = 0
        self.input_data[6] = 0
        self.input_data[7] = 0
        self.input_data[8] = 0
        self.input_data = (self.input_data - self.input_data.mean()) / self.input_data.std()
        # Put the observed state to the NN
        pad = np.zeros((9, 1))
        input_data = np.c_[pad, self.input_data]
        data = torch.from_numpy(input_data.astype(float))
        data = data.view(1, 1, 9, 31)
        self.model.eval()
        probs = self.model(data)
        self.prediction = probs.detach().numpy()
        for i in range(0 , 3):
            self.current_state[2 - i] = self.prediction[0][0][0][self.observed_day - i]
            self.current_state[3 + i] = self.prediction[0][0][0][self.observed_day + i]
            self.current_state[8 - i] = self.simulated_state["Showing symptoms"][0][self.observed_day - i]
        self.current_state[9:12] = self.size
        # Assume we haven't found any 2nd generation case have symptoms at first
        if not return_info:
            return self.current_state
        else:
            return self.current_state, {}

    def step(self, action):
        # Update the input datas
        self.today_result = 0
        self.input_data[0] = 0
        self.input_data[1] = 0
        if self.first_symptom[self.case] != -1:
            self.input_data[0][self.first_symptom[self.case]+1: self.days] = 1
        self.input_data[1][self.observed_day+1: self.days] = 1
        other_symptom = np.delete(self.first_symptom, self.case)
        symptom_num = 0.0
        for i in range (0, self.size - 1):
            if other_symptom[i] <= self.observed_day and other_symptom[i] != -1:
                symptom_num += 1
        self.input_data[2][self.observed_day] = symptom_num
        self.input_data[3] = self.size - 1
        self.input_data[4] = range(0, 30)
        self.input_data[5] = 0
        self.input_data[6] = 0
        self.input_data[7] = 0
        self.input_data[8] = 0
        self.input_data = (self.input_data - self.input_data.mean()) / self.input_data.std()
        pad = np.zeros((9, 1))
        input_data = np.c_[pad, self.input_data]
        data = torch.from_numpy(input_data.astype(float))
        data = data.view(1, 1, 9, 31)
        self.model.eval()
        probs = self.model(data)
        self.prediction = probs.detach().numpy()
        # Update the state
        for i in range(0, 3):
            self.current_state[2 - i] = self.prediction[0][0][0][self.observed_day - i]
            self.current_state[3 + i] = self.prediction[0][0][0][self.observed_day + i]
            self.current_state[8 - i] = self.simulated_state["Showing symptoms"][self.case][self.observed_day - i]
        self.current_state[9:12] = self.size
        sum1 = 0
        sum2 = 0
        sum3 = 0
        # """
        # Threshold
        if self.prediction[0][0][0][self.observed_day] > self.weights:
            action = 1
        else:
            action = 0

        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1 and action == 0:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0 and action == 1:
            sum2 = sum2 + 1
            # """
        """
        # Quarantine based on symptoms
        if self.simulated_state["Showing symptoms"][self.case][self.observed_day - 1] == 0:
            action = 0
        if self.simulated_state["Showing symptoms"][self.case][self.observed_day - 1] == 1:
            action = 1
            
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1 and action == 0:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0 and action == 1:
            sum2 = sum2 + 1
        # """
       
        """
        # 14 days quarantine
        if self.observed_day < self.duration:
            action = 1
        else: 
            if self.simulated_state["Showing symptoms"][self.case][self.observed_day] == 1:
                action = 1
            else:
                action = 0  
                     
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1 and action == 0:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0 and action == 1:
            sum2 = sum2 + 1
        # """
        """
        # CDC Policy
        self.test = 0
        action = 1
        if self.observed_day == 5 and np.all(self.simulated_state["Showing symptoms"][self.case][0:self.observed_day] == 0):
            self.test = 1
            p1 = np.random.multinomial(1, self.confusion_matrix[:,0])
            index1, = np.where(p1 == 1)
            p2 = np.random.multinomial(1, self.confusion_matrix[:,1])
            index2, = np.where(p2 == 1)
            result = [1,0]
            if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1:
                result_5 = result[index1[0]]
            else:
                result_5 = result[index2[0]]
        else: 
            self.test = 0
    
        if self.observed_day >= 8 and self.test ==1 and result_5 == 0 and np.all(self.simulated_state["Showing symptoms"][self.case][0:self.observed_day] == 0):
            action = 0
        if self.observed_day >= 8 and self.test ==1 and result_5 == 1 and self.observed_day <= 14:
            action = 1
        if self.test == 0 and self.observed_day >=11:
            action = 0
        if self.test == 1 and result_5 == 0 and np.any(self.simulated_state["Showing symptoms"][self.case][0:self.observed_day] == 1) and self.observed_day >=11:
            action = 0
        
        if self.test == 1:
            sum3 = sum3 +1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1 and action == 0:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0 and action == 1:
            sum2 = sum2 + 1
        # """
        """
        # Always quarantine
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0:
            sum2 = sum2 + 1
        # """

        """
        # No quarantine
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1:
            sum1 = sum1 + 1
        # """
        self.sum1 = sum1/self.size
        self.sum2 = sum2/self.size
        self.sum3 = sum3/self.size
        arr = np.resize(self.simulated_state["Whether infected"], (self.size, self.days)).astype(float)
        arr = arr[:,3:27]
        infected_days = np.count_nonzero(arr==1)/self.size
        info = [self.sum1, self.sum2, self.sum3, infected_days]
        reward = (-1 * sum1 - self.ratio * sum2 -self.ratio2 * sum3) * 100/self.size
        self.case = self.case + 1
        if self.case == self.size:
            self.observed_day = self.observed_day + 1
            self.case = 0
        done = bool(self.observed_day == self.days-3)
        return self.current_state, reward, done, info
                

    def _simulation(self):
        # Use an array that represents which people get infected. 1 represents get infected.
        self.simulated_state = {
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Whether infected": np.zeros((self.size, self.days))}

        # We assume that the index case has 0.109 probability to be highly transmissive.
        # Under that circumstance, the infectiousness rate becomes 24.4 times bigger.
        flag = bernoulli.rvs(self.p_high_transmissive, size=1)
        if flag == 1:
            self.p_infected = self.p_infected * 24.4
        infected_case = np.array(bernoulli.rvs(self.p_infected, size=self.size))
        for i in range(self.size):
            #  Whether infected
            if infected_case[i] == 1:
                # infected and show symptoms
                if bernoulli.rvs(self.p_symptomatic, size=1) == 1:
                    # Use log normal distribution, mean = 1.57, std = 0.65
                    symptom_day = int(np.random.lognormal(1.57, 0.65, 1)) # day starts to show symptom
                    duration = int(np.random.lognormal(2.70, 0.15, 1))  # duration of showing symptom
                    for j in range(symptom_day, symptom_day + duration):
                        if 0 <= j < self.days:
                            self.simulated_state["Showing symptoms"][i][j] = 1
                    #infected duration
                    period = int(np.random.lognormal(6.67, 2, 1))  # duration of infectiousness
                    for j in range(symptom_day - 2, symptom_day + period):
                        if 0 <= j < self.days:
                            self.simulated_state["Whether infected"][i][j] = 1
                # infected but not showing symptoms
                else:
                    symptom_day = int(np.random.lognormal(1.57, 0.65, 1))
                    period = int(np.random.lognormal(6.67, 2, 1))
                    for j in range(symptom_day - 2, symptom_day + period):
                        if 0 <= j < self.days:
                            self.simulated_state["Whether infected"][i][j] = 1
            # not infected but show some symptoms
            # developing symptoms that is independent of infection status
            symptom_not_infected = bernoulli.rvs(self.p_symptom_not_infected, size=self.days)
            for j in range(self.days):
                if symptom_not_infected[j] == 1:
                    self.simulated_state["Showing symptoms"][i][j] = 1

        if flag == 1:
            self.p_infected = self.p_infected / 24.4
        return self.simulated_state

    def render(self, mode='None'):
        pass

    def close(self):
        pass


