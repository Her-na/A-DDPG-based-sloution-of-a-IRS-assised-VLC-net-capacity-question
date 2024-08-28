import random
import torch
import numpy as np
import copy
from tools import sort_action

class Room_env:
    def __init__(self, M, N, K, M_1, K_1):
        self.M = M  #灯的个数
        self.N = N  #用户数
        self.K = K  #ris个数
        self.M_1 = M_1   #传输的灯的数目
        self.K_1 = K_1   #调制的ris的数目

    def ris_loc(self):
        self.ris_loc = np.zeros((3, self.K))
        for i in range(self.K):

            '''self.ris_loc[:,i] = [3 + 0.25 * (i - int(i/8) * 8), 0, int(i/8) * 0.25 + 1.5]'''
            #self.ris_loc[:, i] = [3 + 0.33 * (i - int(i / 6) * 6), 0, int(i / 6) * 0.2 + 1.8]
            #self.ris_loc[:, i] = [3 + 0.25 * (i - int(i / 8) * 8), 0, 2]
            self.ris_loc[:, i] = [3 + 0.25 * (i - int(i / 8) * 8), 0, int(i / 8) * 0.2 + 1.8]
            '''self.ris_loc[:, i] = [3 + 0.25 * (i - int(i / 8) * 8), 0, 2]'''

    def user_loc(self):
        self.user_loc = np.zeros((3,self.N))
        '''for i in range(self.N):
            user_loc[:,i] = np.array([[],[],[]])
        return user_loc'''
        
        self.user_loc[:, 0] = [2.3, 3.0, 1]
        self.user_loc[:, 1] = [1.8, 4.6, 1]
        self.user_loc[:, 2] = [5.1, 2.1, 1]
        self.user_loc[:, 3] = [4.8, 4.7, 1]
        self.user_loc[:, 4] = [1.8, 3.2, 1]
        
        '''
        self.user_loc[:, 0] = [np.random.rand() * 6, np.random.rand() * 6, np.random.rand() * 3]
        self.user_loc[:, 1] = [np.random.rand() * 6, np.random.rand() * 6, np.random.rand() * 3]
        self.user_loc[:, 2] = [np.random.rand() * 6, np.random.rand() * 6, np.random.rand() * 3]
        self.user_loc[:, 3] = [np.random.rand() * 6, np.random.rand() * 6, np.random.rand() * 3]
        self.user_loc[:, 4] = [np.random.rand() * 6, np.random.rand() * 6, np.random.rand() * 3]
        '''
        
    def LED_loc(self):
        self.LED_loc = np.zeros((3, self.M))
        '''for i in range(self.M):
            LED_loc[:, i] = np.array([[2], [2], [3]])
        return LED_loc'''
        self.LED_loc[:, 0] = [2, 2, 3]
        self.LED_loc[:, 1] = [2, 4, 3]
        self.LED_loc[:, 2] = [4, 2, 3]
        self.LED_loc[:, 3] = [4, 4, 3]


    def gen_Room(self):
        self.ris_loc()
        self.user_loc()
        self.LED_loc()





class RIS_env:
    #传参数
    def __init__(self, Room):
        self.Room = Room
        self.Room.gen_Room()
        #采用方案2
        self.action_dim = self.Room.M + self.Room.K + self.Room.M * self.Room.K + self.Room.K * self.Room.N
        self.state_dim = self.Room.K * self.Room.M * self.Room.N * 2 #+ self.Room.N * self.Room.M
        self.los_gain = np.zeros((self.Room.N, self.Room.M))
        self.nlos_gain = np.zeros((self.Room.N, self.Room.K, self.Room.M))
        self.Check = 0

    #读动作
    def get_action_1(self,action):

        self.L = np.zeros((self.Room.M, 1))
        self.I = np.zeros((self.Room.K, 1))
        self.G = np.zeros((self.Room.K, self.Room.M))
        self.F = np.zeros((self.Room.K, self.Room.N))
        for i in range(self.action_dim):

            if i < self.Room.M:
                self.L[i] = abs(int(np.around(action[0,i])))
            elif i < self.Room.M + self.Room.K:
                self.I[i - self.Room.M] = abs(int(np.around(action[0,i])))
            elif i < self.Room.M + self.Room.K + self.Room.M * self.Room.K:
                s = i - self.Room.M - self.Room.K
                r = int(s / self.Room.M)
                c = s - r * self.Room.M
                self.G[r, c] = abs(int(np.around(action[0,i])))
            elif i < self.Room.M + self.Room.K + self.Room.M * self.Room.K + self.Room.K * self.Room.N:
                s = i - self.Room.M - self.Room.K - self.Room.M * self.Room.K
                r = int(s / self.Room.N)
                c = s - r * self.Room.N
                self.F[r, c] = abs(int(np.around(action[0,i])))

    #优化后的动作采集读取 可以直接读到大部分符合约束的动作
    def get_action_2(self, action):

        self.L = np.zeros((self.Room.M, 1))
        self.I = np.zeros((self.Room.K, 1))
        self.G = np.zeros((self.Room.K, self.Room.M))
        self.F = np.zeros((self.Room.K, self.Room.N))

        index, _ = sort_action(action[0, 0:self.Room.M], self.Room.M_1)
        self.L[index] = 1
        if self.Room.K_1 > 10:
            self.I = np.ones((self.Room.K, 1))
            index, _ = sort_action(action[0, self.Room.M:(self.Room.M + self.Room.K)], self.Room.K - self.Room.K_1)
            self.I[index] = 0
        else:
            index, _ = sort_action(action[0, self.Room.M:(self.Room.M + self.Room.K)], self.Room.K_1)
            self.I[index] = 1
        for row in range(self.Room.K):
            s1 = self.Room.M + self.Room.K + row * self.Room.M
            index1, max1 = sort_action(action[0, s1:(s1 + self.Room.M)], 1)
            if abs(max1) >= 1.8:
                self.G[row, index1] = 0
            else:
                self.G[row, index1] = 1
            s2 = self.Room.M + self.Room.K + self.Room.M * self.Room.K + row * self.Room.N
            index2, max2 = sort_action(action[0, s2:(s2 + self.Room.N)], 1)
            if abs(max2) >= 1.8:
                self.F[row, index2] = 0
            else:
                self.F[row, index2] = 1


    #判断原生的动作是否满足约束
    def constraint_check_1(self):
        self.Check = 0

        if np.sum(self.L, 0) != self.Room.M_1:
            return 0
        self.Check += 1
        if np.sum(self.I, 0) != self.Room.K_1:
            return 0
        self.Check += 1
        if (np.sum(self.G, 1) > 1).any():
            return 0
        self.Check += 1
        if (np.sum(self.F, 1) > 1).any():
            return 0
        self.Check += 1

        #print("通过第一次检查！")

        for m in range(self.Room.M):
            for k in range(self.Room.K):
                if (self.L[m] == 1) & (self.I[k] == 1):
                    if self.G[k, m] != 0:
                        return 0
                if self.I[k] == 1:
                    if np.sum(self.G[k, :]) != 1:
                        return 0
                #改动点 奖励函数
                self.Check += 0.045
                #self.Check += 0.09

        #print("通过第二次检查！")
        return 1

    #距离计算
    def Distance(self):
        d_LED_to_ris = np.zeros((self.Room.K, self.Room.M))
        d_ris_to_user = np.zeros((self.Room.K, self.Room.N))
        d_LED_to_user = np.zeros((self.Room.M, self.Room.N))

        for n in range(self.Room.N):
            for m in range(self.Room.M):
                d_LED_to_user[m, n] = np.sqrt(np.sum((self.Room.LED_loc[:, m] - self.Room.user_loc[:, n]) ** 2, 0))
                for k in range(self.Room.K):
                    d_LED_to_ris[k,m] = np.sqrt(np.sum((self.Room.ris_loc[:, k] - self.Room.LED_loc[:, m]) ** 2, 0))
                    d_ris_to_user[k,n] = np.sqrt(np.sum((self.Room.ris_loc[:, k] - self.Room.user_loc[:, n]) ** 2, 0))

        return d_LED_to_ris, d_ris_to_user, d_LED_to_user

    #信道增益计算
    def Channel_Gains_1(self):
        self.los_gain = np.zeros((self.Room.N, self.Room.M))
        self.nlos_gain = np.zeros((self.Room.N, self.Room.K, self.Room.M))

        #算角度
        d_LED_to_ris, d_ris_to_user, d_LED_to_user = self.Distance()
        los_theta = np.arccos((self.Room.LED_loc[2,:].reshape((self.Room.M,1)) - self.Room.user_loc[2,:]) / d_LED_to_user)
        los_fai = los_theta
        nlos_theta_S = np.arccos((self.Room.LED_loc[2,:] - np.transpose(self.Room.ris_loc[2,:]).reshape((self.Room.K,1))) / d_LED_to_ris)
        nlos_theta_MS = np.arccos((np.transpose(self.Room.ris_loc[2,:]).reshape((self.Room.K,1)) - self.Room.user_loc[2,:]) / d_ris_to_user)
        #UNV_ris = np.zeros((3,self.Room.K))
        UNV_ris = np.array([[0,1,0]])

        #给参数
        A_p = 0.0001#0.05#0.0001#0.0001#0.05
        Lam_index_m = 1
        FoV = np.radians(70)
        g_fai = 1
        refractive_index_n = 1.5
        T_s_fai = np.zeros((self.Room.M, self.Room.N))
        T_s_fai[np.where((los_fai < (FoV/2)) & (los_fai > 0))] = refractive_index_n ** 2 / ((np.sin(FoV/2)) ** 2)


        self.los_gain = ((A_p * (Lam_index_m + 1) / (2 * np.pi * (d_LED_to_user) **2))
                         * np.power(np.cos(los_theta), Lam_index_m)) * T_s_fai * g_fai * np.cos(los_fai)
        self.los_gain = np.transpose(self.los_gain)
        #优化参数位置，减少循环
        for n in range(self.Room.N):
            nlos_theta_D = np.pi / 2 - np.transpose(np.zeros((1,self.Room.M))).reshape((1,self.Room.M)) - nlos_theta_MS[:, n].reshape((self.Room.K,1))
            UVV_ris2user = self.Room.ris_loc - self.Room.user_loc[:, n].reshape((3,1))
            UVV_ris2user = abs(UVV_ris2user)
            T_s = np.zeros((self.Room.K, self.Room.M))
            T_s[np.where((nlos_theta_D < (FoV / 2)) & (nlos_theta_D > 0))] = refractive_index_n ** 2 / ((np.sin(FoV / 2)) ** 2)
            self.nlos_gain[n,:,:] = ((A_p * (Lam_index_m + 1) * np.power(np.cos(nlos_theta_S), Lam_index_m)) / (2 * np.pi *
                             ((np.sqrt((d_LED_to_ris) ** 2 + (d_ris_to_user[:,n]).reshape((self.Room.K,1)) ** 2)) ** 2)) * np.cos(nlos_theta_D) * np.cos(nlos_theta_MS[:,n].reshape((self.Room.K,1)))
                              / np.dot((UNV_ris),UVV_ris2user).reshape((self.Room.K,1)) * T_s * g_fai )

        #print("此函数运行成功")

    #三个信道矩阵的计算，即状态计算
    def Channel_Gains_2(self):
        self.Channel_Gains_1()
        #print(self.los_gain, self.nlos_gain)
        self.nlos_gain_R = np.zeros((self.Room.N, self.Room.K, self.Room.M))
        self.nlos_gain_M = np.zeros((self.Room.N, self.Room.K, self.Room.M))

        for n in range(self.Room.N):
            self.nlos_gain_R[n,:,:] = (np.dot(np.ones((self.Room.K, 1)),np.transpose(self.L)) *
                                  np.dot((np.ones((self.Room.K, 1)) - self.I),np.transpose(np.ones((self.Room.M, 1)))) *
                                  self.G *
                                  np.dot(self.F[:,n].reshape((self.Room.K,1)), np.transpose(np.ones((self.Room.M, 1)))) *
                                  self.nlos_gain[n,:,:])
            self.nlos_gain_M[n, :, :] = (np.dot(np.ones((self.Room.K, 1)), np.transpose(np.ones((self.Room.M, 1)) - self.L)) *
                                    np.dot(self.I,
                                           np.transpose(np.ones((self.Room.M, 1)))) *
                                    self.G *
                                    np.dot(self.F[:, n].reshape((self.Room.K,1)), np.transpose(np.ones((self.Room.M, 1)))) *
                                    self.nlos_gain[n,:, :])

        return self.nlos_gain_R, self.nlos_gain_M

    #生成CDE矩阵
    def gen_CDE(self):
        self.C = np.dot(np.ones((self.Room.N, 1)), np.transpose(self.L)) * np.dot(np.transpose(self.F), self.G)
        self.C[np.where(self.C != 0)] = 1
        #print(self.C)
        self.D = np.dot(np.ones((self.Room.N, 1)), np.transpose(self.I)) * np.transpose(self.F)
        self.E = np.hstack((self.C, self.D))
        #print(self.E)

    #判断CDE矩阵是否满足约束
    def constraint_check_2(self):

        if (np.sum(self.E, 1) < 1).any():
            return 0
        self.Check += 1.2
        if (np.sum(self.E[:,0:4], 0) > 1).any():
            self.Check += np.sum(np.sum(self.E[:,0:4], 0) <= 1) * 0.1
            return 0

        return 1

    #信噪比和码率计算
    def R_SINR(self, nlos_gain_R, nlos_gain_M):
        SINR = np.zeros(self.Room.N)
        R = np .zeros(self.Room.N)
        Noise_reltive = np.zeros(self.Room.N)
        Singal = np.zeros(self.Room.N)
        omega = np.exp(1) / 2 / np.pi
        W = 2e+8#1#2e+6

        #调光和反射系数引入
        #A = np.array([10,10,10,10])
        A = 2.5
        kese = np.array([0.5,0.5,0.5,0.5])
        alpha_ref = 0.9
        var = 1e-21
        #print(self.los_gain)
        for n in range(self.Room.N):
            SINR[n] = (0.25 * ((np.dot(self.C[n,:] * self.los_gain[n,:],np.ones((self.Room.M, 1)) * A) +
                       alpha_ref * (np.transpose(np.ones((self.Room.K, 1)))
                                    @ nlos_gain_R[n,:,:] @
                                    np.ones((self.Room.M , 1))) * A +
                       0.5 * alpha_ref * (np.transpose(np.ones((self.Room.M, 1))) @ np.transpose(nlos_gain_M[n,:,:]) @ np.ones((self.Room.K , 1))* A)) **2)
                       /(np.sum(((np.transpose(self.L) - self.C[n,:]) * self.los_gain[n,:] * A) ** 2) * 0.25 + W * var))
            Noise_reltive[n] = np.sum(((np.transpose(self.L) - self.C[n,:]) * self.los_gain[n,:] * A) ** 2)
            Singal[n] = ((np.dot(self.C[n,:] * self.los_gain[n,:],np.ones((self.Room.M, 1)) * A) +
                       alpha_ref * (np.transpose(np.ones((self.Room.K, 1)))
                                    @ nlos_gain_R[n,:,:] @
                                    np.ones((self.Room.M , 1))) * A +
                       0.5 * alpha_ref * (np.transpose(np.ones((self.Room.M, 1))) @ np.transpose(nlos_gain_M[n,:,:]) @ np.ones((self.Room.K , 1))* A)) **2)

            R[n] = 0.5 * W * np.log2(1 + omega * SINR[n])
        R_total = np.sum(R)
        return R_total * 1e-9

    #交互函数 根据约束检验动作 返回奖励
    def step(self, action):
        reward = -10
        nlos_gain_R = self.nlos_gain_R
        nlos_gain_M = self.nlos_gain_M
        #输入动作
        self.get_action_2(action)
        #self.gen_CDE()
        #nlos_gain_R, nlos_gain_M = self.Channel_Gains_2()
        #检查约束
        if self.constraint_check_1() == 0:
            reward += self.Check * 1
            return reward, nlos_gain_R, nlos_gain_M #, self.los_gain.reshape((1,4,4))
            #return reward, nlos_gain_R, self.los_gain.reshape((1,4,4))
            #return reward, self.E
        #生成CDE
        #print("------------寻找到此处的动作！-------------")
        self.gen_CDE()
        #检查约束
        if self.constraint_check_2() == 0:
            reward += self.Check * 1
            return reward, nlos_gain_R, nlos_gain_M#, , self.los_gainself.los_gain.reshape((1,4,4))
            #return reward, nlos_gain_R, self.los_gain.reshape((1, 4, 4))
            #return reward, self.E
        print("------------寻找到有效动作！-------------")
        #self.gen_CDE()
        #计算新的增益作为next_state
        nlos_gain_R, nlos_gain_M = self.Channel_Gains_2()
        #计算新的reward
        reward = self.R_SINR(nlos_gain_R, nlos_gain_M)
        print(reward)
        return reward, nlos_gain_R, nlos_gain_M#, self.los_gain.reshape((1,4,4))
        #return reward, nlos_gain_R, self.los_gain.reshape((1, 4, 4))
        #return reward, self.E

    #复位函数 训练周期完成复位
    def reset(self):
        action_reset = np.zeros((1, self.Room.M + self.Room.K + self.Room.M * self.Room.K + self.Room.K * self.Room.N))
        #action_reset = np.zeros((1, 2 + self.Room.M + self.Room.K + 2 * self.Room.K))
        '''
        action_reset[0, 0:2] = 1
        action_reset[0, 4:16] = 1
        for k in range(self.Room.K):
            action_reset[0, 38 + k * 4] = 1
        for k in range(self.Room.K):
            action_reset[0, 164 + k * 4] = 1
        '''
        '''
        action_reset[0,0] = 0.5
        action_reset[0,1] = 6/16
        action_reset[0, 2:4] = 1
        action_reset[0, 6:12] = 1
        for k in range(self.Room.K):
            action_reset[0, 22 + k] = 1
            action_reset[0, 38 + k] = 0.3
        '''
        '''
        action_reset[0, 0:2] = 1
        action_reset[0, 4:10] = 1
        action_reset[0, 20] = 1/16
        action_reset[0, 21] = 1/2
        action_reset[0, 22] = 1/16
        action_reset[0, 23] = 1/2
        '''
        '''
        action_reset[0, 0:2] = 1
        action_reset[0, 4:10] = 1
        for k in range(self.Room.K):
            action_reset[0, 22 + k * 4] = 1
        for k in range(self.Room.K):
            action_reset[0, 86 + k * 5] = 1
        '''
        '''
        action_reset[0, 0:2] = 1
        action_reset[0, 4:10] = 1
        for k in range(self.Room.K):
            action_reset[0, 14 + k * 4] = 1
        for k in range(self.Room.K):
            action_reset[0, 46 + k * 5] = 1
        '''
        '''
        action_reset[0, 0:2] = 1
        action_reset[0, 4:7] = 1
        for k in range(self.Room.K):
            action_reset[0, 15 + k * 4] = 1
        for k in range(self.Room.K):
            action_reset[0, 44 + k * 4] = 1
        '''

        action_reset[0, 0:2] = 1
        action_reset[0, 4:10] = 1
        for k in range(self.Room.K):
            action_reset[0, 16 + k * 4] = 1
        for k in range(self.Room.K):
            action_reset[0, 48 + k * 5] = 1


        #print(action_reset)
        self.get_action_2(action_reset)
        print("yes!")
        a = self.constraint_check_1()
        self.gen_CDE()
        b = self.constraint_check_2()
        print(a,b)
        nlos_gain_R, nlos_gain_M = self.Channel_Gains_2()
        #print((np.transpose(self.L) - self.C[0,:]) * self.los_gain[0,:] ** 2)
        r = self.R_SINR(nlos_gain_R,nlos_gain_M)
        print(r)
        #self.E = np.zeros((self.Room.N, (self.Room.M + self.Room.K)))
        #print(nlos_gain_R,nlos_gain_M)
        #return self.E, r, action_reset
        #return nlos_gain_R, nlos_gain_M, r, action_reset
        return nlos_gain_R, nlos_gain_M, r, action_reset
        #return nlos_gain_R, self.los_gain.reshape((1, 4, 4)), r, action_reset









