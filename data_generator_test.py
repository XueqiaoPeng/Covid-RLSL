# -*- coding: utf-8 -*-
# @Author: xueqiao
# @Date:   2022-12-02 13:50:44
# @Last Modified by:   xueqiao
# @Last Modified time: 2023-08-31 21:59:22
from CovidEnv_daily import CovidEnvdaily
# from covidenv import CovidEnv
from csv import writer
import numpy as np
import random


"""
    This code is for generating dataset for Supervise Learning training.
    Use 5*30 matrix to represent the feature of one case
    Index 0: The first day showing the symptom
    Index 1: Whether it is future or (previous and now), 1 represents future and 0 represents (previous and now)
    Index 2: The number of other cases who show symtom
    Index 3: The total number of other cases
    Index 4: Day
    Index 5: Whether we test the case, 1 represents we did test(randomly test 3 days)
    Index 6: Test result
    Index 7: Number of other individuals in the cluster tested that day 
    Index 8: Number of positive results
"""

file_sf = open('./data/feature32.csv', 'w')
writer_sf = writer(file_sf)
file_if = open('./data/infection32.csv', 'w')
writer_if = writer(file_if)

for times in range(0, 50):
    env = CovidEnvdaily()
    full_s = np.resize(env.simulated_state["Showing symptoms"], (env.size, env.days)).astype(float)
    full_i = np.resize(env.simulated_state["Whether infected"], (env.size, env.days)).astype(float)
    sumtest = np.zeros(env.days)
    sumpositive = np.zeros(env.days)
    test = np.zeros((env.size, 30))
    for case in range(0, env.size):
        test[case][:9] = 1
        np.random.shuffle(test[case])
    test_result = np.zeros((env.size, 30))
    feature = np.full((9, env.days),0)
    for day in range(0, env.days):
        for case in range(0, env.size):
            #get first day showing symptoms
            feature[0] = 0
            feature[1] = 0
            idx1 = np.where(full_s[case] == 1)[0] 
            if idx1.size == 0:
                feature[0] = 0
            else:
                feature[0][idx1[0]+1:env.days] = 1
            #present future days, set to 1
            feature[1][day:env.days] = 1
            
            #Keep the first day that other cases show syptom(res)
            other_s = np.delete(full_s,case,0)
            res = np.zeros(other_s.shape)
            idx =  np.arange(res.shape[0])
            args = other_s.astype(bool).argmax(1)
            res[idx, args] = other_s[idx, args]
            res_sum = res.sum(axis = 0)
            #cumulatively get the number of cases showing symptom 
            num = 0
            for i in range(1,env.days):
                if res_sum[i-1] == 1:
                    num = num + 1
                feature[2][i-1 : env.days]= num 
            feature[3, :] = env.size - 1
            feature[4] = range(0, 30)
            feature[5] = test[case]
            C  = np.array([[95,198],[5,9702]])
            confusion_matrix = C / C.astype(np.float).sum(axis=0)
            p1 = np.random.multinomial(1, confusion_matrix[:,0])
            index1, = np.where(p1 == 1)
            p2 = np.random.multinomial(1, confusion_matrix[:,1])
            index2, = np.where(p2 == 1)
            result = [1,0]
            if feature[5][day]== 1 and env.simulated_state["Whether infected"][case][day] == 1:
                feature[6][day] = result[index1[0]]
                test_result[case][day] = feature[6][day]

            if feature[5][day] == 1 and env.simulated_state["Whether infected"][case][day] == 0:
                feature[6][day] = result[index2[0]]
                test_result[case][day] = feature[6][day]
            if feature[5][day] == 0:
                feature[6][day] = 0

            feature[7][day] = sumtest[day - 1] - test[case][day - 1]
            feature[8][day] = sumpositive[day -1] -  test_result[case][day - 1]
            feature[7][0] = 0
            feature[8][0] = 0
            feature = (feature - feature.mean()) / feature.std()
            # print("\n")
        
            # print(feature)
            # print(env.simulated_state["Whether infected"][case])
        
            writer_sf.writerows(feature.astype(float))
            writer_if.writerow(full_i[case])

        sumtest[day] = np.sum(test, axis= 0)[day]
        sumpositive[day] = np.sum(test_result, axis= 0)[day]
            

# file_sf.close()
# file_if.close()


