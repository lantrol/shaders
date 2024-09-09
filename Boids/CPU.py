import pygame
from math import sin, cos, pi, sqrt
import random

pygame.init()
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
NUM_BIRDS = 800
SCALE = 0.5
PADDING = 50
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 24, True)

class Grid():
    def __init__(self, birds):
        self.birds = birds
        self.cellSize = SCREEN_HEIGHT/9
        self.numRows = 9
        self.grid = {}

    def updateGrid(self):
        self.grid = {}
        for bird in self.birds:
            hash = bird.xpos//self.cellSize + (bird.ypos//self.cellSize)*self.numRows
            if self.grid.get(hash) == None:
                self.grid[hash] = [bird]
            else:
                self.grid[hash].append(bird)
            bird.gridCell = hash
    def getCell(self, hash):
            look = [hash+self.numRows, hash-self.numRows, hash+1, hash-1]
            check_birds = self.grid.get(hash)
            for check in look:
                if self.grid.get(int(check)) != None:
                    check_birds = check_birds+self.grid.get(int(check))
            return check_birds
    def draw(self):
        for i in range(self.numRows):
            pygame.draw.line(screen, (80, 80, 80), (0, i*self.cellSize), (SCREEN_WIDTH, i*self.cellSize))
        for i in range(self.numRows):
            pygame.draw.line(screen, (80, 80, 80), (i*self.cellSize, 0), (i*self.cellSize, SCREEN_HEIGHT))



class Bird():
    def __init__(self, screen, grid, birds, pos, vel, draw_range = False):
        self.screen = screen
        self.birds = birds
        self.grid = grid
        self.xpos = pos[0]
        self.ypos = pos[1]
        self.vel = 2*SCALE
        self.xvel = vel[0]
        self.yvel = vel[1]
        self.size = 6*SCALE
        self.v_range = 110*SCALE
        self.avoid_range = 40*SCALE
        self.avoidfactor = 0.03
        self.matchfactor = 0.02
        self.centerfactor = 0.02
        self.turnfacotor = 0.05
        self.draw_range = draw_range
        self.gridCell = 0
    def draw(self):
        if self.draw_range:
            pygame.draw.circle(self.screen, (87, 230, 144), (self.xpos, self.ypos), self.v_range)
            pygame.draw.circle(self.screen, (87, 230, 218), (self.xpos, self.ypos), self.avoid_range)
        pygame.draw.circle(self.screen, (0, 0, 255), (self.xpos, self.ypos), self.size)
        pygame.draw.line(self.screen, (200, 0, 0), (self.xpos, self.ypos), (self.xpos+self.xvel*20, self.ypos+self.yvel*20), 2)
    def step(self):
        matchvx, matchvy, centervx, centervy, avoidvx, avoidvy = self.get_vector()
        self.xvel += matchvx*self.matchfactor
        self.yvel += matchvy*self.matchfactor
        self.xvel += centervx*self.centerfactor
        self.yvel += centervy*self.centerfactor
        self.xvel += avoidvx*self.avoidfactor
        self.yvel += avoidvy*self.avoidfactor

        #Check if outside screen
        if self.xpos < PADDING:
            self.xvel += self.turnfacotor
        if self.xpos > SCREEN_WIDTH - PADDING:
            self.xvel -= self.turnfacotor
        if self.ypos < PADDING:
            self.yvel += self.turnfacotor
        if self.ypos > SCREEN_HEIGHT - PADDING:
            self.yvel -= self.turnfacotor

        vLength = sqrt(self.xvel**2 + self.yvel**2)
        self.xvel, self.yvel = self.xvel/vLength, self.yvel/vLength
        self.xpos += self.xvel*self.vel
        self.ypos += self.yvel*self.vel
    def get_vector(self):
        matchvx = 0
        matchvy = 0
        centervx = 0
        centervy = 0
        avoidvx = 0
        avoidvy = 0
        n_birds = 0
        avoid_birds = 0
        check_birds = self.grid.getCell(self.gridCell)
        for b in check_birds:
            #match
            dist = sqrt((self.xpos-b.xpos)**2+(self.ypos-b.ypos)**2)
            if dist > self.v_range:
                continue
            n_birds += 1
            matchvx += b.xvel
            matchvy += b.yvel
            #Center
            centervx += b.xpos
            centervy += b.ypos
            if dist < self.avoid_range and b != self and dist != 0:
                avoidvx += self.xpos - b.xpos
                avoidvy += self.ypos - b.ypos
                avoid_birds += 1

        #Match
        matchvx, matchvy = matchvx/n_birds, matchvy/n_birds
        vLength = sqrt(matchvx*matchvx + matchvy*matchvy)
        if vLength == 0:
            matchvx, matchvy = 0, 0
        else:
            matchvx, matchvy = matchvx/vLength, matchvy/vLength
        #Center
        centervx, centervy = centervx/n_birds, centervy/n_birds
        centervx = centervx - self.xpos
        centervy = centervy - self.ypos
        vLength = sqrt(centervx*centervx + centervy*centervy)
        if vLength == 0:
            centervx, centervy = 0, 0
        else:
            centervx, centervy = centervx/vLength, centervy/vLength
        #Avoid
        if avoid_birds != 0:
            avoidvx, avoidvy = avoidvx/avoid_birds, avoidvy/avoid_birds
            vLength = sqrt(avoidvx**2 + avoidvy**2)
            avoidvx, avoidvy = avoidvx/vLength, avoidvy/vLength
        return matchvx, matchvy, centervx, centervy, avoidvx, avoidvy

birds = []
grid = Grid(birds)
for i in range(NUM_BIRDS):
    pos = (random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT))
    angle = random.random()*2*pi
    vel = (cos(angle), sin(angle))
    birds.append(Bird(screen, grid, birds, pos, vel))
birds[0].draw_range = True
print(len(birds))

running = True
dt=1
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #Game loop
    grid.updateGrid() #Update grid lookup
    for bird in birds:
        bird.step()

    #Draw cicle
    screen.fill((155, 155, 155))
    grid.draw()
    for bird in birds:
        bird.draw()
    fps = font.render(f"{1/dt:.2f}", True, "green")
    screen.blit(fps, (0,0))
    pygame.display.flip()

    dt = clock.tick(60)/1000
pygame.quit()
