import torch
import numpy as np

class model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, output_dim),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

class value_model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(value_model, self).__init__()

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, output_dim)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


class REINFORCE:
    def __init__(self, input_dim, output_dim, lr_policy, lr_value, baseline_mode=1):
        self.policy_model = model(input_dim, output_dim)
        self.optimizer_policy = torch.optim.Adam(self.policy_model.parameters(), lr=lr_policy)
        self.baseline_mode = baseline_mode
        
        if self.baseline_mode == 2:
            self.value_model = value_model(input_dim, 1)
            self.optimizer_value = torch.optim.Adam(self.value_model.parameters(), lr=lr_value)

        if self.baseline_mode == 1:
            self.past_cum_rew = []

    def ask(self, state):
        input = torch.Tensor(state)
        output = self.policy_model.forward(input)
        return self.sample_from_dist(output.tolist())
    
    def sample_from_dist(self, dist):
        action_probability_dist = [p/sum(dist) for p in dist]
        action = np.random.choice([0, 1,2,3], 1, p=action_probability_dist)
        return action[0]
    
    def train_on_trajectorys(self, trajectorys):
        if self.baseline_mode == 0:
            self.train_vanilla_REINFORCE(trajectorys)
        if self.baseline_mode == 1:
            self.train_avg_baseline_REINFORCE(trajectorys)
        if self.baseline_mode == 2:
            self.train_learned_baseline_REINFORCE(trajectorys)
    
    def train_vanilla_REINFORCE(self, trajectorys):
        loss_policy = torch.Tensor([0.0])
        for trajectory in trajectorys:
            cumulative_reward = torch.tensor(sum([x.reward for x in trajectory]))
            for p in trajectory:
                # update policy model
                probability_dist = self.policy_model(torch.Tensor(p.state))
                prob_sum = torch.sum(probability_dist)
                probability_dist = probability_dist/prob_sum 
                loss_policy += torch.log(probability_dist[p.action])*cumulative_reward
        loss_policy *= -1/len(trajectorys)
        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()

    def train_avg_baseline_REINFORCE(self, trajectorys):
        loss_policy = torch.Tensor([0.0])
        for trajectory in trajectorys:
            cumulative_reward = torch.tensor(sum([x.reward for x in trajectory]))
            # for avg baseline
            if len(self.past_cum_rew) > 100:
                self.past_cum_rew.pop(0)
            self.past_cum_rew.append(cumulative_reward)
            baseline = sum(self.past_cum_rew)/len(self.past_cum_rew)
            for p in trajectory:
                advantage = cumulative_reward - baseline
                # update policy model
                probability_dist = self.policy_model(torch.Tensor(p.state))
                prob_sum = torch.sum(probability_dist)
                probability_dist = probability_dist/prob_sum 
                loss_policy += torch.log(probability_dist[p.action])*advantage
        loss_policy *= -1/len(trajectorys)
        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()
        

    def train_learned_baseline_REINFORCE(self, trajectorys):
        loss_policy = torch.Tensor([0.0])
        loss_value = torch.Tensor([0.0])
        for trajectory in trajectorys:
            cumulative_reward = torch.tensor(sum([x.reward for x in trajectory]))
            for p in trajectory:
                # update value model
                value_estimate = self.value_model(torch.Tensor(p.state))
                expectation_offset = cumulative_reward - value_estimate
                loss_value += 0.5*(expectation_offset)**2
                # update policy model
                probability_dist = self.policy_model(torch.Tensor(p.state))
                prob_sum = torch.sum(probability_dist)
                probability_dist = probability_dist/prob_sum 
                loss_policy += torch.log(probability_dist[p.action])*expectation_offset.detach()
        loss_policy *= -1/len(trajectorys)
        loss_value *= 1/len(trajectorys)
        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        loss_policy.backward()
        loss_value.backward() 
        self.optimizer_policy.step()
        self.optimizer_value.step()
        