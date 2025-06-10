import circenv.mob
import dql
import numpy as np
import munchhausen_dql as mdql

class dql_agent(circenv.mob.mob):
    def __init__(self, x, y, r):
        super().__init__(x, y, r)
        #self.learner = dql.dql(input_dim=10, output_dim=6, gamma = 0.9999999, epsilon=0.2, buffer_size=500000, lr=0.00001, copy_interval=2000)
        self.input_dim = 9
        self.stm_length = 4
        self.total_input_dim = self.stm_length*self.input_dim
        self.learner = mdql.munchhausen_dql(input_dim=self.total_input_dim, output_dim=6, gamma = 0.99999, epsilon=0.5, buffer_size=20000, lr=0.0001, copy_interval=2000, tau=0.2, alpha=0.9)
        self.observation_buffer = []
        self.last_action = None
        self.view_ray_angles = [-0.9, -0.4, 0, 0.4, 0.9]
        self.step = 0
        
        for i in range(self.stm_length+2):
            self.observation_buffer.append([0.0 for j in range(self.input_dim)])

    def reset(self):
        for i in range(self.stm_length+2):
            self.observation_buffer.append([0.0 for j in range(self.input_dim)])
        self.step = 0

    def act(self):
        #act
        if len(self.observation_buffer) == 0:
            self.observation_buffer.append(self.observe())

        input = self.craft_input(1)
        self.last_action = self.learner.ask(input)
        self.action(self.last_action)
 
    def reward(self, reward, terminated):
        #new observation
        self.step+=1
        self.append_to_obs_buffer(self.observe())
        self.learner.replay_buffer.append(self.craft_input(2), self.craft_input(1), self.last_action, reward, terminated)
        
    def append_to_obs_buffer(self, observation):
        if len(self.observation_buffer) > 10:
            self.observation_buffer.pop(0)
        self.observation_buffer.append(observation)

    def observe(self):
        observation = []
        for res in self.ray_cast_hits:
            if len(res) > 0:
                vec = np.array([self.pos_x-res[0][0], self.pos_y-res[0][1]])
                observation.append(float(np.linalg.norm(vec)))
            else:
                observation.append(float(-1))
        observation.append(float(self.pos_x/500))
        observation.append(float(self.pos_y/500))
        #observation.append(float(self.vel_x))
        #observation.append(float(self.vel_y))
        observation.append(float(self.rotation/(2*np.pi)))
        observation.append(self.step/400)
        return observation
    
    def craft_input(self, step):
        input = []
        for i in range(step, step+self.stm_length):
            #print(i)
            input += self.observation_buffer[-i]
        #print(len(input))
        return input
