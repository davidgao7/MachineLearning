# develop a flappy phoenix using pygame and NN!
# reference: https://github.com/techwithtim/NEAT-Flappy-Bird/blob/master/flappy_bird.py
# TODO: 2. know what variable AI should control in the game
# TODO: 3. design a algorithm and perform learning!

# TASK 1. display the game
from __future__ import print_function
import pygame # when run, use <python3> instead of <python>
import os
import time
import random
import neat

# initialize the pygame
pygame.init()
# create pygame screen
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

# load frames
DISPLAY_WINDOW = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])  # set mode need to be before converting images

BirdIMG = [
    pygame.transform.scale2x(
        pygame.image.load(
            os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bird"+str(x)+".png")
        )
    )for x in range(1,4) # easier way to write following three
    # pygame.transform.scale2x(
    #     pygame.image.load(os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bird1.png"))),
    # pygame.transform.scale2x(
    #     pygame.image.load(os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bird2.png"))),
    # pygame.transform.scale2x(
    #     pygame.image.load(os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bird3.png")))
]

SUBIMG = {
    "Pipe_frame": pygame.transform.scale2x(
        pygame.image.load(
            os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/pipe.png")).convert_alpha()),
    "Base_frame": pygame.transform.scale2x(
        pygame.image.load(
            os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/base.png")).convert_alpha()),
    "Background_frame": pygame.transform.scale2x(
        pygame.image.load(
            os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bg.png")).convert_alpha())
}

def update_screen():
    pygame.display.update()

def draw_bird_window(game_window, bird):
    game_window.blit(SUBIMG["Background_frame"], (0, 0))
    bird.animate(game_window)


def draw_pipe_window(game_window, pipe):
    game_window.blit(SUBIMG["Background_frame"], (0, 0))
    pipe.animate(game_window)


def draw_base_window(game_window, base):
    game_window.blit(SUBIMG["Background_frame"], (0, 0))
    base.animate(game_window)


def rotate_img(img, surf, topLeft, angle):
    rotated_img = pygame.transform.rotate(img, angle)
    new_rect = rotated_img.get_rect(center=img.get_rect(topleft=topLeft).center)
    surf.blit(rotated_img, new_rect.topleft)


# for later AI bird generations updating, we need a container to hold info which need tobe updated
class Bird:

    MAX_ROTATION = 25  # how much bird tilt each move
    ROTATE_VELOCITY = 20  # how fast the bird img rotate each frame
    ANIMATION_TIME = 5  # how fast the bird will flap its wings in the frames
    MAX_VISIBLE_PIXIELS_HEIGHT = 16  # maximum vertial range you can see the game

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0  # how much the image will tilt for each frame on the screen
        self.time = 0
        self.velocity = 0
        self.img_count = 0
        self.img = BirdIMG[0]
        self.bird_vel = 3
        self.alive = True

    def jump(self):
        self.velocity = -10.5  # since going down is negative
        self.time = 0

    def move(self):
        self.time += 1  # track the time
        d = self.velocity * self.time + 0.5 * self.bird_vel * self.time ** 2  # how the bird move(recall free fall equation: delta x = v0*t + 0.5*a*t^2) plus sign since pixel(0,0) start at the top left

        if self.y + d >= Bird.MAX_VISIBLE_PIXIELS_HEIGHT:  # once user makes bird to jump to the top of the screen and still want to jump, keep the bird in the same height
            self.y = Bird.MAX_VISIBLE_PIXIELS_HEIGHT
            self.tilt = Bird.TILT_DEGREES  # tilt up

        elif self.y < 0:  # "player" not handling the bird , bird die
            self.alive = False
            self.y -= 2
        
        self.y = self.y + d
        
        if d < 0 or self.y < self.y + 50: #tilt up
            if self.tilt < Bird.MAX_ROTATION:
                self.tilt = Bird.MAX_ROTATION

        elif self.tilt > -90:
            self.tilt -= Bird.ROTATE_VELOCITY

    def animate(self, game_window):
        self.img_count += 1

        # fly animation (determine what img need to show according to counter)
        self.img = BirdIMG[self.img_count % 3]

        # if bird is in free fall, wings are not flapping
        if self.tilt == -90:
            self.img = BirdIMG[1]

        self.move()
        
        # actual rotation for the bird, rotate bird around its center
        rotate_img(
            img=self.img,
            surf=game_window,
            topLeft=(self.x, self.y),
            angle=self.tilt
        )

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe():
    
    GAP = 200 # the gap between top pipe and bottom pipe
    VEL = 5 # how fast the pipe moving from right to left of screen
    
    def __init__(self,x):
        self.x = x
        self.height = 0
        
        self.top_len = 0
        self.bottom_len = 0
        
        # flip the bottom pipe img vertically
        self.PIPE_TOP = pygame.transform.flip(SUBIMG["Pipe_frame"],False,True) # flip(Surface, xbool, ybool)
        self.PIP_BOTTOM = SUBIMG["Pipe_frame"]
        
        # inital pipe height
        self.set_height()

    
    def set_height(self):
        # generate top pipe and bottom pipe pair with different gap
        self.height = random.randrange(50, 450)
        self.bottom_len = self.height - self.PIPE_TOP.get_height()
        self.top_len = self.height + Pipe.GAP
    
    def move(self):
        # move pipe with given velocity
        self.x -= Pipe.VEL # every pipe moves left with the same velocity
    
    def animate(self, window):
        # similar to how we draw bird, we draw random pipe on both sides of up/down screen
        window.blit(self.PIPE_TOP, (self.x, self.top_len)) # for drawing the top pipe
        window.blit(self.PIP_BOTTOM, (self.x, self.bottom_len)) # for drawing the bottom pipe

    # IMPORTANT: now we need to consider the colision, we need a flag to signal once 
    # a bird die
    def colide(args):
        pass

class Base:
    
    VEL = 5 # since Base and Pipe are on rest frame (need to same velocity as pipe)
    WIDTH = SUBIMG["Base_frame"].get_width() # need to fill the world with ground...
    
    def __init__(self, y):
        self.y = y
        self.left = 0
        self.right = Base.WIDTH # since base not actually moving
    
    def move(self): # scroll the image to express displacement, same with pipe
        self.left  -= Base.VEL # since base move from right to left
        self.right -= Base.VEL
        
        # 1 cycle done, need redraw
        if self.left + Base.WIDTH < 0: # left is out of scope
            self.left = self.right + Base.WIDTH
        if self.right + Base.WIDTH < 0: # right is out of scope
            self.right = self.left + Base.WIDTH
    
    def animate(self, window):
        window.blit(SUBIMG["Base_frame"], (self.left,self.y))
        window.blit(SUBIMG["Base_frame"], (self.right,self.y))

def run(path):
    bird = Bird(x=200, y=200)
    base = Base(730)
    pipes = [Pipe(700), Pipe(500), Pipe(300), Pipe(700)]
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        # TODO: pipe move not working
        # generate multiple pipes
        for pipe in pipes:
            # if the pipe are off the screen, put it in trash and remove all in once
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                del(pipe)
            else:
                pipe.move() # update pipe position
                draw_pipe_window(DISPLAY_WINDOW, pipe) # draw each pipe
        
        # base.move() # update base position (act like velocity)
        # draw_bird_window(DISPLAY_WINDOW, bird)
        # draw_base_window(DISPLAY_WINDOW, base) # update base
        update_screen() # update the screen after drawing everything


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-FeedForward.txt")  # parameter need to tune
    run(config_path)
    
    