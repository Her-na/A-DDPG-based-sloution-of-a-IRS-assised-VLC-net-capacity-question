import RIS_env
import numpy as np
import torch
import tools
from DDPG import Agent_DDPG
from DQN import Agent_DQN
from RIS_env import RIS_env, Room_env
import matplotlib.pyplot as plt

#设定参数
M = 4
N = 5      #4
K = 16
M_1 = 3
K_1 = 5

hidden_dim = 880#2080#1210#880#980#1200#1980#2000#1600#980#1600#980#1260#6425#2400#2020一层#820#512#820#512#1250#800
sigma = 0.08#0.1#0.01#0.08#0.01#0.35 0.15
gamma = 0.98#0.98#0.95#0.97
actor_lr = 1e-4#1e-3#1e-3#1e-4#1e-4#5e-3#4e-5
critic_lr = 1e-3#1e-2#1e-2#1e-3#1e-3#4e-2#3e-4
tau = 0.0001#0.000001#0.00001#0.0001#0.0001#0.00000005#0.005
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

episodes = 1000
max_steps = 100
ep_reward_list = []
ep_loss_list = []
avg_score = np.zeros(episodes)
avg_loss = np.zeros(episodes)
action_save = []

#设定环境:DDPG
env_Room = Room_env(M, N, K, M_1, K_1)
env = RIS_env(env_Room)
agent = Agent_DDPG(hidden_dim, sigma, actor_lr, critic_lr, tau, gamma, device, env)
set = 1

for i in range(episodes):
    if i <= 1:
        states = np.zeros((1, env.state_dim))
        #以下为动作采样3、4的语句
        #next_states, reward, action = env.reset()
        #next_states = tools.get_vector2(next_states)
        next_states_1, next_states_2, reward, action = env.reset()
        next_states = tools.get_vector(next_states_1, next_states_2)
    if i > 1:

        states, action, reward, next_states = agent.memory.sample_buffer(1)
        action = action.reshape((1, agent.action_dim))
        next_states = next_states.reshape(agent.state_dim)
        states = states.reshape(agent.state_dim)

    for i0 in range(max_steps):

        states = next_states
        action = agent.take_action(states, set, i)
        reward, next_states_1, next_states_2 = env.step(action)
        if reward > 0:
            print("有效动作！")
        next_states = tools.get_vector(next_states_1, next_states_2)
        #reward, next_states = env.step(action)
        #next_states = tools.get_vector2(next_states)
        if reward < 0:
            if i < 1:
                agent.remember(states, action, reward, next_states)
            elif np.random.rand() >= 0.85:
                agent.remember(states, action, reward, next_states)
        else:
            agent.remember(states, action, reward, next_states)

        if (i < 2) & (i0 < 100):
            transition_dict = tools.dict_gen(states, action, next_states, reward)
        else:
            states_buff, action_buff, reward_buff, next_states_buff = agent.memory.sample_buffer(64)
            transition_dict = tools.dict_gen(states_buff, action_buff, next_states_buff, reward_buff)
            agent.update(transition_dict)
            loss = agent.mean_loss.detach()
            ep_loss_list.append(loss)
        ep_reward_list.append(reward)
        set = np.mean(ep_reward_list[-2:])
    avg_score[i] = np.mean(ep_reward_list[-100:])
    avg_loss[i] = np.mean(ep_loss_list[-100:])
    print('episode', i, 'loss %.1f' % avg_loss[i], 'avg score %.1f' % avg_score[i])


'''
#设定环境:DQN
env_Room = Room_env(M, N, K, M_1, K_1)
env = RIS_env(env_Room)
agent = Agent_DQN(hidden_dim, actor_lr, gamma, 0.2, 100, device, env)

for i in range(episodes):
    if i <= 1:
        states = np.zeros((1, env.state_dim))
        next_states_1, next_states_2, reward, action = env.reset()
        next_states = tools.get_vector(next_states_1, next_states_2)
    if i > 1:
        states, action, reward, next_states = agent.memory.sample_buffer(1)
        action = action.reshape((1, agent.action_dim))
        next_states = next_states.reshape(agent.state_dim)
        states = states.reshape(agent.state_dim)
        
    for i0 in range(max_steps):
        states = next_states
        action = agent.take_action(next_states)
        action = np.array(action)
        action = action.reshape((1,env.action_dim))
        #print(action)
        reward, next_states_1, next_states_2 = env.step(action)
        next_states = tools.get_vector(next_states_1, next_states_2)
        
        if reward < 0:
            if i < 1:
                agent.remember(states, action, reward, next_states)
            elif np.random.rand() >= 0.95:
                agent.remember(states, action, reward, next_states)
        else:
            agent.remember(states, action, reward, next_states)

        if (i < 2) & (i0 < 100):
            transition_dict = tools.dict_gen(states, action, next_states, reward)
        else:
            states_buff, action_buff, reward_buff, next_states_buff = agent.memory.sample_buffer(32)
            transition_dict = tools.dict_gen(states_buff, action_buff, next_states_buff, reward_buff)
            agent.update(transition_dict)
        ep_reward_list.append(reward)
    avg_score[i] = np.mean(ep_reward_list[-100:])
    print('episode', i, 'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])
'''

plt.plot([i for i in range(episodes)], avg_score, linewidth=1, label="M=4, L=5, K=16, M$^{'}$ =2, K$^{'}$ =6, P=5")
plt.legend(loc='lower right', fontsize=7)
plt.xlabel("episode")
plt.ylabel("Sum Rate (Gbps)")
#plt.grid(b=True, which='major')
#plt.grid(b=True, which='minor', alpha=0.4)
plt.show()
np.save(r"C:\Users\\Desktop\Save_DDPG_new_8", avg_score)


