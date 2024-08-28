import numpy as np
import torch
import random
import collections
from DNN import DQN_Qnet
import torch.nn.functional as F
from tools import ReplayBuffer

class Agent_DQN:
    def __init__(self, hidden_dim, learning_rate, gamma,
                 epsilon, target_update, device, env):
        self.env = env
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        self.Value_net = DQN_Qnet(self.state_dim, hidden_dim, self.action_dim).to(device)
        # 目标网络
        self.target_Value_net = DQN_Qnet(self.state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.Value_net.parameters(),lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        
        self.memory = ReplayBuffer(100000, self.state_dim, self.action_dim)
        self.batch_size = 32
        self.mean_loss = 0

    def take_action(self, state):
        with torch.no_grad():# epsilon-贪婪策略采取动作
            state = torch.unsqueeze(torch.tensor([state], dtype=torch.float).to(self.device),0)
        if np.random.random() > self.epsilon:
            q = self.Value_net(state)
            action = q.argmax(dim=-1).tolist()
        else:
            #print("yes!")
            action = np.random.randint(0,2,size = (1, self.action_dim))
            #print(action)

        return action

    def remember(self, state, action, reward, next_state):
        self.memory.store_transition(state ,action ,reward, next_state)

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.Value_net(state).max().detach()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)

        q_values = self.Value_net(states).gather(1, actions.reshape((32,164,1)))
        q_values = torch.sum(q_values,1)
        max_action = self.Value_net(next_states).max(-1)[1].view(32,-1)
        max_next_q_values = self.target_Value_net(next_states).gather(1,max_action.reshape((32,164,1)))
        max_next_q_values = torch.sum(max_next_q_values,1)
        q_targets = rewards + self.gamma * max_next_q_values
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_Value_net.load_state_dict(self.Value_net.state_dict())
        self.count += 1
