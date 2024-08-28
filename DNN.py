import torch
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):#设置几层?
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = torch.nn.LayerNorm(100)
        self.fc3 = torch.nn.Linear(100, hidden_dim)
        self.ln3 = torch.nn.LayerNorm(hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.ln1(x)
        #x = self.dropout(x)
        #x = F.relu(self.fc2(x))
        #x = self.ln2(x)
        #x = self.dropout(x)
        #x = F.relu(self.fc3(x))
        #x = self.ln3(x)
        #x = self.dropout(x)

        return torch.tanh(self.fc4(x))


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 200)
        self.ln2 = torch.nn.LayerNorm(300)
        self.fc3 = torch.nn.Linear(300, hidden_dim)
        self.ln3 = torch.nn.LayerNorm(hidden_dim)
        self.fc_out = torch.nn.Linear(200, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, a):
        cat = torch.cat([x, a],dim=1)
        #print("cat结果为：",cat)
        x = F.relu(self.fc1(cat))
        #x = self.ln1(x)
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.ln2(x)
        #x = self.dropout(x)
        #x = F.relu(self.fc3(x))
        #x = self.ln3(x)
        #x = self.dropout(x)
        
        return self.fc_out(x)


class DQN_Qnet(torch.nn.Module):
    def __init__(self, state_dim , hidden_dim, action_dim):
        super(DQN_Qnet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 800)
        self.fc2 = torch.nn.Linear(800, 200)
        self.fc3 = torch.nn.Linear(200, 800)
        self.fc4 = torch.nn.Linear(800, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.head = torch.nn.ModuleList()
        for i in range(action_dim):
         self.head.append(torch.nn.Linear(hidden_dim, 2))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        q_values = [h(x) for h in self.head]
        return torch.stack(q_values,dim=0)


