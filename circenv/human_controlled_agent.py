import circenv.mob
import pygame
import numpy as np

class human_controlled_agent(circenv.mob.mob):
    def __init__(self, x, y, r):
        super().__init__(x, y, r)
        self.view_ray_angles = [-0.1, 0, 0.1]
    
    def act(self):
        keys=pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.action(0)
        if keys[pygame.K_a]:
            self.action(1)
        if keys[pygame.K_s]:
            self.action(2)
        if keys[pygame.K_d]:
            self.action(3)
        if keys[pygame.K_j]:
            self.action(4)
        if keys[pygame.K_k]:
            self.action(5)

        #prep observations
        observation = []
        for res in self.ray_cast_hits:
            if len(res) > 0:
                vec = np.array([self.pos_x-res[0][0], self.pos_y-res[0][1]])
                observation.append(np.linalg.norm(vec))
            else:
                observation.append(-1)
        observation.append(self.pos_x/500)
        observation.append(self.pos_y/500)
        observation.append(self.vel_x)
        observation.append(self.vel_y)
        observation.append(self.rotation/(2*np.pi))
        print(observation)