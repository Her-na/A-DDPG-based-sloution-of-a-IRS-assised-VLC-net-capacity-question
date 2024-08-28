import torch
import torch.nn.functional as F
from DNN import PolicyNet, ValueNet
from tools import ReplayBuffer
from tools import noise_mask

class Agent_DDPG:
    def __init__(self, hidden_dim, sigma, actor_lr, critic_lr, tau, gamma, device, env):
        self.env = env
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        self.actor = PolicyNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.critic = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        self.critic_2 = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        self.target_actor = PolicyNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_critic = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        self.target_critic_2 = ValueNet(self.state_dim, hidden_dim, self.action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.device = device

        self.memory = ReplayBuffer(100000, self.state_dim, self. action_dim)
        self.batch_size = 64
        self.mean_loss = 0

    def take_action(self, state, set, eposide):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        #这里获取数值 多个数据的张量使用detach
        action = self.actor(state).detach()
        if set < 0:
            #noise_add = noise(self.env.Room.M, self.env.Room.K, self.env.Room.N,10)
            #action = action + 0.02 * np.random.randn(self.action_dim)
            #action = action + noise_mask(0.15)
            action = action + noise_mask(0.3)
        elif set > 0:
            print("there!")
            #action = action + 0.001 * np.random.randn(self.action_dim)
            #noise_add = noise(self.env.Room.M, self.env.Room.K, self.env.Room.N,1)
            #action = action + self .sigma * np.random.randn(self.action_dim)
            action = action + noise_mask(self.sigma)
        #action = torch.clip_(action,0 ,1)
        #print(action)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def remember(self, state, action, reward, next_state):
        self.memory.store_transition(state ,action ,reward, next_state)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        #actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        #rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        action_new = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, action_new)
        q_targets = rewards + self.gamma * next_q_values

        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.mean_loss = torch.mean(critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = - torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)









