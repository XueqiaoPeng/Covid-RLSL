# -*- coding: utf-8 -*-
# @Author: xueqiao
# @Date:   2023-02-09 12:50:33
# @Last Modified by:   xueqiao
# @Last Modified time: 2023-11-12 11:39:28
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import bernoulli
from .Supervised_Learning_test import NeuralNetwork
import torch
#test daily

class CovidEnvdaily(gym.Env):

    def __init__(self):
        self.size = np.random.randint(20,40,1)[0]
        # self.size = 32
        self.days = 30  # Assume we observe the 2nd generation for 30 days
        # We assume weight_infect_no_quarantine = -1, and calculate weight_no_infect_quarantine from the ratio
        self.ratio = 0.2
        self.weights = self.ratio/(1 + self.ratio)
        self.ratio2 = 0.1
        self.p_high_transmissive = 0.109  # Probability that the index case is highly transmissive
        self.p_infected = 0.03  # Probability that a person get infected (given index case is not highly transmissive)
        self.p_symptomatic = 0.8  # Probability that a person is infected and showing symptom
        self.p_symptom_not_infected = 0.01  # Probability that a person showing symptom but not infected
        self.observed_day = 3  # The agent starts to get involved in day 3
        self.duration = 14  # The default quarantine duration is 14
        self.test = 1
        self.today_result = None
        self.first_symptom = None
        self.simulated_state =  self._simulation()
        self.input_data = None
        self.prediction = None
        self.current_state = None
        self.test_result = None
        self.all_test = None
        C  = np.array([[34,22],[14,1869]])
        self.confusion_matrix = C / np.sum(C , axis=0)
        # Initialize the model
        self.model = NeuralNetwork().double()
        self.sum1 = None
        self.sum2 = None
        self.sum3 = None
        self.model.load_state_dict(torch.load('/users/PCON0023/xueqiao/miniconda3/envs/spinningup/lib/python3.6/site-packages/gym/envs/CovidRL/model/model_822.pth'))
        self.case = None
        """
        Use a long vector to represent the observation. Only focus on one 2nd generation.
        We assume that person get exposed at day 0. 
        Index 0 to 2 represent the prediction of whether infected during three days before the observing day. 
        Index 3 to 5 represent the prediction of whether infected during three days after the observing day. 
        Index 6 to 8 represent whether that person shows symptoms in recent three days.
        Index 9 to 11 represents whether testing
        Index 12 to 14 represents the result of testing
        Index 15 to 17 represents how many tests ran in past three days
        Index 18 to 20 represents how many tests ran in past three days
        Index 21 to 23 represents the cluster size
        """
        self.observation_space = spaces.Box(low=0, high=50, shape=(24,), dtype=np.float32)

        """
        0 for test and no quarantine, 1 for test and quarantine
        2 for test positive and quarantine, test negative and no quarantine
        
        """
        self.action_space = spaces.Discrete(3)

        self.seed()
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # We start to trace when the 1st generation case is tested positive.
    def reset(self, return_info=False):
        # self.size = np.random.randint(20,40,1)[0]
        self.observed_day = 3
        self.duration = 14
        self.test = 1
        self.case = 0
        self.simulated_state = self._simulation()
        # Initialize the current state
        self.current_state = np.zeros(24)
        # Find when first show symptom
        self.first_symptom = np.full(self.size, -1)
        self.case_test = np.zeros((self.size, self.days))
        self.case_result = np.zeros((self.size, self.days))
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
        self.current_state[21:24] = self.size
        # Assume we haven't found any 2nd generation case have symptoms at first
        if not return_info:
            return self.current_state
        else:
            return self.current_state, {}

    def step(self, action):
        # Update the input data
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
        p1 = np.random.multinomial(1, self.confusion_matrix[:,0])
        index1, = np.where(p1 == 1)
        p2 = np.random.multinomial(1, self.confusion_matrix[:,1])
        index2, = np.where(p2 == 1)
        result = [1,0]
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1:
                self.today_result = result[index1[0]]
        else:
                self.today_result = result[index2[0]]
        self.case_test[self.case][self.observed_day] = self.test
        self.case_result[self.case][self.observed_day] = self.today_result
        self.input_data[5][self.observed_day] = self.test
        self.input_data[6][self.observed_day] = self.today_result
        self.input_data[7][self.observed_day] = self.sumtest[self.observed_day-1] - self.case_test[self.case][self.observed_day-1]
        self.input_data[8][self.observed_day] = self.sumpositive[self.observed_day-1] - self.case_result[self.case][self.observed_day-1]
        self.input_data = (self.input_data - self.input_data.mean()) / self.input_data.std()
        # Put the updated observed state to the NN
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
            self.current_state[11 - i] = self.case_test[self.case][self.observed_day - i]
            self.current_state[14 - i] = self.case_result[self.case][self.observed_day -i]
            self.current_state[17 - i] = self.sumtest[self.observed_day - i -1]
            self.current_state[20 - i] = self.sumpositive[self.observed_day - i -1]
        self.current_state[21:24] = self.size
        sum1, sum2, sum3 = 0, 0, 0
        # """
        # RL
        if self.test == 1:
            sum3 = sum3 +1

        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1 and action == 0:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1 and (action == 2 and self.today_result == 0 ):
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0 and action == 1:
            sum2 = sum2 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0 and (action == 2 and self.today_result == 1 ):
            sum2 = sum2 + 1
        # """
        reward = (-1 * sum1 - self.ratio * sum2 - self.ratio2 * sum3) * 100/self.size
        self.case = self.case + 1
        if self.case == self.size - 1:
            self.sumtest[self.observed_day] = np.sum(self.case_test, axis= 0)[self.observed_day]
            self.sumpositive[self.observed_day] = np.sum(self.case_result, axis= 0)[self.observed_day]
            self.observed_day = self.observed_day + 1
            self.case = 0
        done = bool(self.observed_day == self.days - 3)
        return self.current_state, reward, done, {}


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
                    #  Whether infected
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
