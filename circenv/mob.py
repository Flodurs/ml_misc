import numpy as np

class mob:
    def __init__(self, x, y, r):
        self.pos_x = x
        self.pos_y = y
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.acceleration = 2.0
        self.reibung = 0.95
        self.radius = r
        self.view_ray_angles = []
        self.view_ray_len = 600
        self.ray_cast_hits = []
        self.rotation = 0.5
    
    def update(self, world):
        self.pos_x += self.vel_x
        self.pos_y += self.vel_y
        if self.pos_x < 0:
            self.pos_x = 0
        if self.pos_x > world.size_x:
            self.pos_x = world.size_x
        if self.pos_y < 0:
            self.pos_y = 0
        if self.pos_y > world.size_y:
            self.pos_y = world.size_y
        self.vel_x*=self.reibung
        self.vel_y*=self.reibung
        if abs(self.vel_y) <= 0.0001:
            self.vel_y = 0
        if abs(self.vel_x) <= 0.0001:
            self.vel_x = 0
        self.calc_view_ray_collisions(world)
        self.act()

    def act(self):
        pass

    def action(self, a):
        acc = np.array([0.0, 0.0])
        if a == 0:
            acc = np.array([1.0,0.0])
        if a == 1:
            acc = np.array([0.0,-1.0])
        if a == 2:
            acc = np.array([-1.0,0.0])
        if a == 3:
            acc = np.array([0.0,1.0])
        if a == 4:
            self.rotation += 0.05
        if a == 5:
            self.rotation -= 0.05
        acc*=self.acceleration
        vel=np.matmul(np.array([[np.cos(self.rotation),np.sin(self.rotation)], [-np.sin(self.rotation), np.cos(self.rotation)]]), acc)
        #vel = self.normalize(vel)*self.acceleration
        self.vel_x += vel[0]
        self.vel_y += vel[1]
        self.rotation = np.fmod(self.rotation, 2*np.pi)
        if self.rotation < 0:
            self.rotation += 2*np.pi
        
    def calculate_view_ray_direction(self):
        dirs = []
        for angle in self.view_ray_angles:
            dirs.append([np.cos(angle-self.rotation), np.sin(angle-self.rotation)])
        return dirs
    
    def calc_view_ray_collisions(self, world):
        dirs = self.calculate_view_ray_direction()
        self.ray_cast_hits = []
        for dir in dirs:
            self.ray_cast_hits.append(world.raycast(self.pos_x+dir[0]*(self.radius+1), self.pos_y+dir[1]*(self.radius+1), self.pos_x+dir[0]*self.view_ray_len, self.pos_y+dir[1]*self.view_ray_len))
    
    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def normalize_angle(self, v, s, e):
        width = e-s
        offset = v-s
        return (offset-(np.floor(offset/width)*width))+s
            