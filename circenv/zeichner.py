import pygame 

class zeichner:
    def __init__(self):

        self.screenSize = (800, 800)
        self.screen = pygame.display.set_mode(self.screenSize)
        self.xOffset = 10
        self.yOffset = 10
        self.canvas = pygame.Surface((500,500))
        
    def render(self, world):
        self.canvas.fill((0, 0, 0))
        self.screen.fill((0, 20, 0))
        self.draw_mobs(world.mobs)
        self.screen.blit(self.canvas, (self.xOffset, self.yOffset))
        pygame.display.flip()

    def draw_mobs(self, mobs):
        for mob in mobs:
            pygame.draw.circle(self.canvas, (0, 0, 255), [mob.pos_x, mob.pos_y], mob.radius)
            dirs = mob.calculate_view_ray_direction()
            for index, dir in enumerate(dirs):
                if len(mob.ray_cast_hits[index]) > 0:
                    pygame.draw.line(self.canvas, (255, 0, 0) , [mob.pos_x, mob.pos_y], [mob.ray_cast_hits[index][0][0], mob.ray_cast_hits[index][0][1]], 1)
                else:
                    pygame.draw.line(self.canvas, (255, 0, 0) , [mob.pos_x, mob.pos_y], [mob.pos_x+dir[0]*mob.view_ray_len, mob.pos_y+dir[1]*mob.view_ray_len], 1)