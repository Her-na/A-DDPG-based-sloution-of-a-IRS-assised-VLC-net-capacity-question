import numpy as np
import copy
import torch

#用于向量化
def get_vector(M1, M2):
    page1, row1, col1 = M1.shape
    page2, row2, col2 = M2.shape
    M1 = np.reshape(M1, page1 * row1 * col1)
    M2 = np.reshape(M2, page2 * row2 * col2)
    M = np.append(M1, M2)
    return M

def get_vector2(M1):
    row1, col1 = M1.shape
    M1 = np.reshape(M1,row1 * col1)
    return M1

#用于生成字典
def dict_gen(states, actions, next_states, rewards):
    dict_1 = {}
    dict_1['states'] = states
    dict_1['actions'] = actions
    dict_1['next_states'] = next_states
    dict_1['rewards'] = rewards
    return dict_1


#排序
def sort_action(action, num):

    data = copy.deepcopy(action)
    index = []
    if type(data) is np.ndarray:
        max_num = np.max(data)
    else:
        max_num = torch.max(data)
    for n in range(num):
        index.append(np.argmax(data))
        data[np.argmax(data)] = -1e10

    return index, max_num


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, new_state):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]

        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        return states, actions, rewards, states_


def noise_mask(sigma):
    noise = []
    noise = np.append(noise, sigma * np.random.randn(4 + 12))
    noise = np.append(noise, 1 * sigma * np.random.randn(4 * 12))
    noise = np.append(noise, 1 * sigma * np.random.randn(5 * 12))
    
    return noise
