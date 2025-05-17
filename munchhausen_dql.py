import torch
import numpy as np
import random

class model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, output_dim)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
    
class transition:
    def __init__(self, state_a, state_b, action, reward, terminal):
        self.state_a = state_a
        self.state_b = state_b
        self.action = action
        self.reward = float(reward)
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

class munchhausen_dql:
    def __init__(self, input_dim, output_dim, gamma, epsilon, buffer_size, lr, copy_interval, tau, alpha):
        self.model_prim = model(input_dim, output_dim)
        self.target = model(input_dim, output_dim)
        self.target.load_state_dict(self.model_prim.state_dict())
        self.replay_buffer = replay_buffer(buffer_size)
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = torch.optim.Adam(self.model_prim.parameters(), lr=lr)
        self.loss_func = torch.nn.MSELoss()
        self.loss_func = torch.nn.HuberLoss(reduction='mean', delta=1.0)

        self.step = 0
        self.copy_interval = copy_interval
        self.tau = tau
        self.alpha = alpha

    def ask(self, state):
        sm = torch.nn.Softmax(dim=0)
        if torch.bernoulli(torch.tensor([self.epsilon])) == 0:
            return self.sample_from_dist(sm(self.model_prim(torch.tensor(state))/self.tau).detach().tolist())
        return np.random.choice([x for x in range(self.output_dim)])
        
    def train_step(self):
        self.step+=1
        self.eposilon_decay_lin()
        #print(len(self.replay_buffer.transitions))
        sm = torch.nn.Softmax(dim=0)
        for i in range(30):
            transition = self.replay_buffer.sample_transition()
            if transition.terminal:
                y = transition.reward
                y = torch.tensor(y)
            else:
                factor = 0
                q_vals = self.target(torch.tensor(transition.state_b))
                dist = sm(q_vals/self.tau)
                #log_dist = torch.clip(self.tau*log_sm(q_vals/self.tau),min=-1,max=0)
                v = torch.max(q_vals)
                logsum = torch.logsumexp((q_vals - v)/self.tau, dim=0)
                log_dist = q_vals - v - self.tau*logsum
                log_dist = torch.clip(log_dist, min=-1, max=0)
                for j in range(self.output_dim):
                    factor += dist[j]*(q_vals[j] - log_dist[j]) 
                y = transition.reward + self.alpha*log_dist[transition.action] + self.gamma*factor
            #print(y)
            loss = self.loss_func(y, self.model_prim.forward(torch.tensor(transition.state_a))[transition.action])
            loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.step%self.copy_interval == 0:
            self.target.load_state_dict(self.model_prim.state_dict())
            print("---------------------------------------------------------Update")
    
    def eposilon_decay_lin(self):
        summand = (0.5) / (50000)
        self.epsilon = max(self.epsilon - summand, 0.0000001)
        if self.step%100 == 0:
            print(self.epsilon)
        #print(self.epsilon)
    
    def sample_from_dist(self, dist):
        #print("ac dist")
        action_probability_dist = [p/sum(dist) for p in dist]
        #print(action_probability_dist)
        action = np.random.choice([x for x in range(self.output_dim)], 1, p=action_probability_dist)
        return action[0]