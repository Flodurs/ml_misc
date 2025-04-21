import circenv.human_controlled_agent
import circenv.boulder
import circenv.dql_agent
import matplotlib.pyplot as plt
import numpy as np
import random

class world:
    def __init__(self):
        self.mobs = []
        self.size_x = 500
        self.size_y = 500
        self.mobs.append(circenv.boulder.boulder(self.size_x/2, self.size_y/2, 50))
        #self.mobs.append(circenv.human_controlled_agent.human_controlled_agent(40, 40, 10))
        self.mobs.append(circenv.dql_agent.dql_agent(250, 250, 10))
        self.step_num = 0
        self.episode_step = 0
        self.fig, self.ax = plt.subplots()
        self.plot_data = []
        self.rewards = []
        plt.ion()
        plt.show()
        
    def update(self):
        self.step_num+=1
        self.episode_step+=1
        for mob in self.mobs:
            mob.update(self)
            vec = np.array([self.mobs[0].pos_x-self.mobs[1].pos_x, self.mobs[0].pos_y-self.mobs[1].pos_y])
            dist = np.linalg.norm(vec)
            vec = self.normalize(vec)
            self.mobs[0].pos_x -= vec[0]*0.15
            self.mobs[0].pos_y -= vec[1]*0.15
            if isinstance(mob, circenv.dql_agent.dql_agent):
                if dist < self.mobs[0].radius+self.mobs[1].radius:
                    mob.reward(-1.0, True)
                    self.rewards.append(0.0)
                    self.reset()
                else:
                    mob.reward(1.0, False)
                    self.rewards.append(0.1)
                if self.episode_step > 10000:
                    self.rewards.append(0.0)
                    mob.reward(0.1, True)
                    self.reset()
        if self.step_num%15 == 0:
            self.mobs[1].learner.train_step()

            
    def reset(self):
        n = 1
        self.mobs[n].pos_x = 250
        self.mobs[n].pos_y = 250
        self.mobs[n].vel_x = 0.0
        self.mobs[n].vel_y = 0.0
        self.mobs[n].rotation = random.uniform(0, 2*np.pi)
        self.mobs[n].observation_buffer = []
        self.mobs[n].last_action = None
        #self.mobs[n].ray_cast_hits = []
        self.mobs[0].pos_x = random.randrange(0, 500)
        self.mobs[0].pos_y = random.randrange(0, 500)
        self.episode_step = 0

        self.plot_data.append(sum(self.rewards))
        self.rewards = []
        self.ax.clear()
        self.ax.plot(self.plot_data)
        self.fig.canvas.draw()


    def raycast(self, x0, y0, x1, y1):
        hits = []
        for mob in self.mobs:
            intersections = self.line_circle_intersect(x0, y0, x1, y1, mob.pos_x, mob.pos_y, mob.radius)
            if len(intersections) > 0:
                hits.append([intersections[0], intersections[1]])
        return hits

    def line_circle_intersect(self, x0, y0, x1, y1, x2, y2, r):
        #print(f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, r: {r}")
        a = (x1-x0)**2+(y1-y0)**2
        b = 2*((x0-x2)*(x1-x0)+(y0-y2)*(y1-y0))
        c = (x0-x2)**2+(y0-y2)**2-r**2
        discriminant = b**2-4*a*c
        #print(f"discriminant:{discriminant}")
        if discriminant < 0.0:
            return []
        root = discriminant**0.5
        if discriminant == 0.0:
            t = (-b-root)/(2*a)
            if t >= 0 and t <= 1:
                xi = (x1-x0)*t+x0
                yi = (y1-y0)*t+y0
                return [xi, yi]
            else:
                return []
        t0 = (-b-root)/(2*a)
        t1 = (-b+root)/(2*a)
        #print(f"t0: {t0}")
        #print(f"t1: {t1}")
        flag_t0 = (t0 >= 0 and t0 <= 1)
        flag_t1 = (t1 >= 0 and t1 <= 1)
        #print(flag_t0)
        #print(flag_t1)
        if flag_t0 and flag_t1:
            xi0 = (x1-x0)*t0+x0
            yi0 = (y1-y0)*t0+y0
            xi1 = (x1-x0)*t1+x0
            yi1 = (y1-y0)*t1+y0
            return [xi0, yi0, xi1, yi1]
        if flag_t0:
            xi0 = (x1-x0)*t0+x0
            yi0 = (y1-y0)*t0+y0
            return [xi0, yi0]
        if flag_t1:
            xi1 = (x1-x0)*t1+x0
            yi1 = (y1-y0)*t1+y0
            return [xi1, yi1]
        return []

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
    
#w = world()
#print(w.line_circle_intersect(1,-1,-4,-4,1,-1,1))