# develop a flappy phoenix using pygame and NN!

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
        pygame.image.load(os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bird1.png"))),
    pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bird2.png"))),
    pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bird3.png")))
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


def draw_bird_window(game_window, bird):
    game_window.blit(SUBIMG["Background_frame"], (0, 0))
    bird.animate(game_window)
    pygame.display.update()

def draw_pipe_window(game_window, pipe):
    game_window.blit(SUBIMG["Background_frame"], (0, 0))
    pipe.set_height()
    pipe.animate(game_window)
    pygame.display.update()

def rotate_img(img, surf, topLeft, angle):
    rotated_img = pygame.transform.rotate(img, angle)
    new_rect = rotated_img.get_rect(center=img.get_rect(topleft=topLeft).center)
    surf.blit(rotated_img, new_rect.topleft)


# for later AI bird generations updating, we need a container to hold info which need tobe updated
class Bird:
    
    TILT_DEGREES = 25  # how much bird tilt each move
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
            if self.tilt < Bird.TILT_DEGREES:
                self.tilt = Bird.TILT_DEGREES
        
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


def run(path):
    bird = Bird(x=200, y=200)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        
        draw_window(DISPLAY_WINDOW, bird)
        draw_bird_window(DISPLAY_WINDOW, bird)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-FeedForward.txt")  # parameter need to tune
    run(config_path)

import time
import random
import neat