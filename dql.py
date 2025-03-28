import torch
import numpy as np
import random

class model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, output_dim),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
    
class transition:
    def __init__(self, state_a, state_b, action, reward, terminal):
        self.state_a = state_a
        self.state_b = state_b
        self.action = action
        self.reward = reward
        self.terminal = terminal

class replay_buffer:
    def __init__(self, size):
        self.transitions = []
        self.size = size
    
    def append(self, state_a, state_b, action, reward, terminal):
        if len(self.transitions) >= self.size:
            self.transitions.pop(0)
        self.transitions.append(transition(state_a, state_b, action, reward, terminal))

    def sample_transition(self):
        return random.choice(self.transitions)

class dql:
    def __init__(self, input_dim, output_dim, gamma, epsilon, buffer_size, lr, copy_interval):
        self.model_prim = model(input_dim, output_dim)
        self.target = model(input_dim, output_dim)
        self.target.load_state_dict(self.model_prim.state_dict())
        self.replay_buffer = replay_buffer(buffer_size)
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = torch.optim.Adam(self.model_prim.parameters(), lr=lr)
        self.loss_func = torch.nn.MSELoss()
        self.step = 0
        self.copy_interval = copy_interval

    def ask(self, state):
        if torch.bernoulli(torch.tensor([self.epsilon])) == 0:
            return torch.argmax(self.model_prim.forward(torch.tensor(state))).item()
        return np.random.choice([x for x in range(self.output_dim)])
        
    def train_step(self):
        self.step+=1
        for i in range(20):
            transition = self.replay_buffer.sample_transition()
            if transition.terminal:
                y = transition.reward
                y = torch.tensor(y)
            else:
                y = transition.reward + self.gamma*max(self.target.forward(torch.tensor(transition.state_b)))
            print(y)
            
            loss = self.loss_func(y, self.model_prim.forward(torch.tensor(transition.state_a))[transition.action])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.step%self.copy_interval == 0:
            self.target.load_state_dict(self.model_prim.state_dict())

    def train_step_ddqn(self):
        self.step+=1
        for i in range(64):
            transition = self.replay_buffer.sample_transition()
            if transition.terminal:
                y = float(transition.reward)
                y = torch.tensor(y)
            else:
                y = transition.reward + self.gamma*self.target.forward(torch.tensor(transition.state_b))[torch.argmax(self.model_prim.forward(torch.tensor(transition.state_b)))]
            print(y)
            loss = self.loss_func(y, self.model_prim.forward(torch.tensor(transition.state_a))[transition.action])
            loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.step%self.copy_interval == 0:
            self.target.load_state_dict(self.model_prim.state_dict())