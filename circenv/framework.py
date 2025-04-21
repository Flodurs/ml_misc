import pygame
import circenv.world
import circenv.zeichner

class framework:
    def __init__(self):
        self.bRunning = True 
        self.welt = circenv.world.world()
        self.zeichner = circenv.zeichner.zeichner()
        pygame.init()
        
    def run(self):
        while self.bRunning == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.bRunning = False
            self.update()
            self.render()
            
    def update(self):
        self.welt.update()
        
    def render(self):
        self.zeichner.render(self.welt)
        
f = framework()
f.run()