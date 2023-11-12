# -*- coding: utf-8 -*-
# @Author: xueqiao
# @Date:   2023-02-09 12:50:33
# @Last Modified by:   xueqiao
# @Last Modified time: 2023-11-12 11:39:40
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import bernoulli
#pure SL

class CovidEnvpureRL(gym.Env):

    def __init__(self):
        # self.size = 32
        self.size = np.random.randint(2,40,1)[0]
        self.days = 30  # Assume we observe the 2nd generation for 30 days
        # We assume weight_infect_no_quarantine = -1, and calculate weight_no_infect_quarantine from the ratio
        self.ratio = 0.1
        self.weights = self.ratio/(1 + self.ratio)
        self.ratio2 = 0
        self.p_high_transmissive = 0.109  # Probability that the index case is highly transmissive
        self.p_infected = 0.03  # Probability that a person get infected (given index case is not highly transmissive)
        self.p_symptomatic = 0.8  # Probability that a person is infected and showing symptom
        self.p_symptom_not_infected = 0.01  # Probability that a person showing symptom but not infected
        self.observed_day = 3  # The agent starts to get involved in day 3
        self.duration = 14  # The default quarantine duration is 14
        self.test = 0  # test for RL at first
        self.first_symptom = None
        self.simulated_state = None
        self.current_state = None
        self.case_test = None
        self.case_result = None
        C  = np.array([[34,22],[14,1869]])
        self.confusion_matrix = C / C.astype(np.float).sum(axis=0)
        self.case = None
        self.sum1 = 0
        self.sum2 = 0
        self.sum3 = 0

        """
        Use a long vector to represent the observation. Only focus on one 2nd generation.
        1 means showing symptoms. 0 means not showing symptoms. -1 means unobserved.
        We assume that person get exposed at day 0. 
       
        Index 0 to 2 represent whether that person shows symptoms in recent three days.
        Index 3 to 5 represents whether testing
        Index 6 to 8 represents the result of testing
        Index 9 to 11 represents how many tests ran in past three days
        Index 12 to 14 represents how many positive tests in past three days
        Index 15 to 17 represents the cluster size
        """
        self.observation_space = spaces.Box(low=0, high=50, shape=(18,), dtype=np.float32)

        """
        0 for no test no quarantine, 1 for no test and quarantine
        2 for test and no quarantine, 3 for test and quarantine
        4 for test positive and quarantine, test negative and no quarantine
        """
        self.action_space = spaces.Discrete(2)

        self.seed()
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # We start to trace when the 1st generation case is tested positive.
    def reset(self, return_info=False):
        # self.size = np.random.randint(2,40,1)[0]
        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()
        # Initialize the current state
        self.current_state = np.zeros(18)
        self.observed_day = 3
        self.test = 0
        self.case = 0
        self.case_test = np.zeros((self.size, self.days))
        self.case_result = np.zeros((self.size, self.days))
        self.sumtest = np.zeros(self.days)
        self.sumpositive = np.zeros(self.days)
        for i in range(0, 3):
            self.current_state[2 - i] = self.simulated_state["Showing symptoms"][0][self.observed_day - i]
        self.current_state[15:18] = self.size
        # Assume we haven't found any 2nd generation case have symptoms at first
        if not return_info:
            return self.current_state
        else:
            return self.current_state, {}

    def step(self, action):
        # Update the state from the result of simulation
        self.today_result = 0
        if action == 2 or action == 3 or action == 4:
            self.test = 1
        else:
            self.test = 0
        p1 = np.random.multinomial(1, self.confusion_matrix[:,0])
        index1, = np.where(p1 == 1)
        p2 = np.random.multinomial(1, self.confusion_matrix[:,1])
        index2, = np.where(p2 == 1)
        result = [1,0]
        if self.test == 1:
            if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1:
                self.today_result = result[index1[0]]
            else:
                self.today_result = result[index2[0]]
        else:
            self.today_result = 0
        self.case_test[self.case][self.observed_day] = self.test
        self.case_result[self.case][self.observed_day] = self.today_result

        for i in range(0, 3):
            self.current_state[2 - i] = self.simulated_state["Showing symptoms"][self.case][self.observed_day - i]
            self.current_state[5 - i] = self.case_test[self.case][self.observed_day - i]
            self.current_state[7 - i] = self.case_result[self.case][self.observed_day -i]
            self.current_state[11 - i] = self.sumtest[self.observed_day - i -1]
            self.current_state[14 - i] = self.sumpositive[self.observed_day - i -1]
        self.current_state[15:18] = self.size
        sum1, sum2, sum3 = 0, 0, 0

        """
        # RL
        
        if self.test == 1:
            sum3 = sum3 + 1

        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1 and (action == 0 or action== 2):
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1 and (action == 4 and self.today_result == 0):
            sum1 = sum1 + 1
            
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0 and (action == 1 or action == 3):
            sum2 = sum2 + 1
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 0 and (action == 4 and self.today_result == 1):
            sum2 = sum2 + 1
        # """ 
        # """
        # No quarantine
        if self.simulated_state["Whether infected"][self.case][self.observed_day] == 1:
            sum1 = sum1 + 1
        # """
                  
        reward = (-1 * sum1 - self.ratio * sum2 - self.ratio2 * sum3) * 100/self.size
        
        self.case = self.case + 1
        if self.case == self.size - 1:
            self.sumtest[self.observed_day] = np.sum(self.case_test, axis= 0)[self.observed_day]
            self.sumpositive[self.observed_day] = np.sum(self.case_result, axis= 0)[self.observed_day]
            self.observed_day = self.observed_day + 1
            self.case = 0
        self.sum1 = sum1
        self.sum2 = sum2
        self.sum3 = sum3
        done = bool(self.observed_day == self.days - 3)
        return self.current_state, reward, done, {}

    def _simulation(self):
        # Use an array that represents which people get infected. 1 represents get infected.
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
