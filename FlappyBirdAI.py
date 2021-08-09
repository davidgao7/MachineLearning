# develop a flappy phoenix using pygame and NN!
# reference: https://github.com/techwithtim/NEAT-Flappy-Bird/blob/master/flappy_bird.py
# TODO: 2. know what variable AI should control in the game
# TODO: 3. design a algorithm and perform learning!

# TASK 1. display the game
from __future__ import print_function

import os
import random

import neat
import pygame  # when run, use <python3> instead of <python>

# initialize the pygame
pygame.font.init()
# create pygame screen
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 800
FLOOR = 730
START_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = True
# load frames
DISPLAY_WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))  # set mode need to be before converting images
pygame.display.set_caption("Flappy Bird")

BirdIMG = [
    pygame.transform.scale2x(
        pygame.image.load(
            os.path.join("imgs", "/Users/tengjungao/PycharmProjects/MachineLearning/bird" + str(x) + ".png")
        )
    ) for x in range(1, 4)  # easier way to write following three
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

DISPLAY_WINDOW.blit(SUBIMG["Background_frame"], (0, 0))

def rotate_img(img, surf, topLeft, angle):
    rotated_img = pygame.transform.rotate(img, angle)
    new_rect = rotated_img.get_rect(center=img.get_rect(topleft=topLeft).center)
    surf.blit(rotated_img, new_rect.topleft)


bird_generation = 0


# for later AI bird generations updating, we need a container to hold info which need tobe updated
class Bird:
    # all birds will have these:
    MAX_ROTATION = 25  # how much bird tilt each move
    ROTATE_VELOCITY = 20  # how fast the bird img rotate each frame
    ANIMATION_TIME = 5  # how fast the bird will flap its wings in the frames
    MAX_VISIBLE_PIXIELS_HEIGHT = 16  # maximum vertical range you can see the game

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0  # how much the image will tilt for each frame on the screen
        self.time = 0
        self.velocity = 0
        self.img_count = 0
        self.img = BirdIMG[0]
        self.bird_vel = 3

    def jump(self):
        self.velocity = -10.5  # since going down is negative
        self.time = 0
        self.height = self.y
        self.move()

    def move(self):
        self.time += 1  # track the time

        # downard displacement
        d = self.velocity * self.time + 0.5 * self.bird_vel * (
                self.time ** 2)  # how the bird move(recall free fall equation: delta x = v0*t + 0.5*a*t^2) plus sign since pixel(0,0) start at the top left

        if d >= Bird.MAX_VISIBLE_PIXIELS_HEIGHT:  # once user makes bird to jump to the top of the screen and still want to jump, keep the bird in the same height
            d = (d / abs(d)) * Bird.MAX_VISIBLE_PIXIELS_HEIGHT

        elif self.y < 0:  # "player" not handling the bird , bird die
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.y + 50:  # tilt up
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
    GAP = 200  # the gap between top pipe and bottom pipe
    VEL = 5  # how fast the pipe moving from right to left of screen

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top_len = 0
        self.bottom_len = 0
        self.passed = False

        # flip the bottom pipe img vertically
        self.PIPE_TOP = pygame.transform.flip(SUBIMG["Pipe_frame"], False, True)  # flip(Surface, xbool, ybool)
        self.PIP_BOTTOM = SUBIMG["Pipe_frame"]

        # inital pipe height
        self.set_height()

    def set_height(self):
        # generate top pipe and bottom pipe pair with different gap
        self.height = random.randrange(50, 450)
        self.top_len = self.height - self.PIPE_TOP.get_height()
        self.bottom_len = self.height + Pipe.GAP

    def move(self):
        # move pipe with given velocity
        self.x -= Pipe.VEL  # every pipe moves left with the same velocity

    def animate(self, window):
        # similar to how we draw bird, we draw random pipe on both sides of up/down screen
        window.blit(self.PIPE_TOP, (self.x, self.top_len))  # for drawing the top pipe
        window.blit(self.PIP_BOTTOM, (self.x, self.bottom_len))  # for drawing the bottom pipe

    # IMPORTANT: now we need to consider the colision, we need a flag to signal once
    # a bird die
    def colide(self, bird, window):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIP_BOTTOM)
        top_offset = (self.x - bird.x, self.top_len - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom_len - round(bird.y))

        b_pt = bird_mask.overlap(bottom_mask, bottom_offset)
        t_pt = bird_mask.overlap(top_mask, top_offset)

        if b_pt or t_pt:
            return True

        return False


class Base:
    VEL = 5  # since Base and Pipe are on rest frame (need to same velocity as pipe)
    WIDTH = SUBIMG["Base_frame"].get_width()  # need to fill the world with ground...

    def __init__(self, y):
        self.y = y
        self.left = 0
        self.right = Base.WIDTH  # since base not actually moving

    def move(self):  # scroll the image to express displacement, same with pipe
        self.left -= Base.VEL  # since base move from right to left
        self.right -= Base.VEL

        # 1 cycle done, need redraw
        if self.left + Base.WIDTH < 0:  # left is out of scope
            self.left = self.right + Base.WIDTH
        if self.right + Base.WIDTH < 0:  # right is out of scope
            self.right = self.left + Base.WIDTH

    def animate(self, window):
        window.blit(SUBIMG["Base_frame"], (self.left, self.y))
        window.blit(SUBIMG["Base_frame"], (self.right, self.y))


def draw_bird_window(game_window, bird):
    # game_window.blit(SUBIMG["Background_frame"], (0, 0))
    bird.animate(game_window)


def draw_pipe_window(game_window, pipe):
    # game_window.blit(SUBIMG["Background_frame"], (0, 0))
    pipe.animate(game_window)


def draw_base_window(game_window, base):
    # game_window.blit(SUBIMG["Background_frame"], (0, 0))
    base.animate(game_window)


def draw_window(win, birds, pipes, base, score, pipe_indx):
    win.blit(SUBIMG["Background_frame"], (0, 0))

    for pipe in pipes:  # include pipes
        pipe.animate(win)

    base.animate(win)  # include base

    # draw lines from bird to pipe
    for bird in birds:
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255, 0, 0),
                                 (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2),
                                 (pipes[pipe_indx].x + pipes[pipe_indx].PIPE_TOP.get_width() / 2,
                                  pipes[pipe_indx].height),
                                 5)
                pygame.draw.line(win, (255, 0, 0),
                                 (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2),
                                 (pipes[pipe_indx].x + pipes[pipe_indx].PIP_BOTTOM.get_width() / 2,
                                  pipes[pipe_indx].bottom_len), 5)
            except:
                pass
        bird.animate(win)

        # score
        score_label = START_FONT.render("Score: " + str(score), 1, (255, 255, 255))  # white
        win.blit(score_label, (10, 10))

        # alive
        score_label = START_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
        win.blit(score_label, (10, 50))

        pygame.display.update()


def eval_genomes(genomes, config):
    """
    simulation of the flappy bird game, update weight base on distance reach during
    the game
    """
    WIN = DISPLAY_WINDOW
    # can be used outside of function
    win = WIN

    # start group of birds
    nets = []
    birds = []
    ge = []

    # NEAT: Neuroevolution of Augmenting Topologies
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_indx = 0
        if len(birds) > 0:
            # determine whether to use the first or second pipe on the screen for neural network input
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_indx = 1

            for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
                ge[x].fitness += 0.1
                bird.move()

                # send bird location. top pipe location and bottom pipe location and determine from network whether to jump or not
                output = nets[birds.index(bird)].activate(
                    (bird.y, abs(bird.y - pipes[pipe_indx].height), abs(bird.y - pipes[pipe_indx].bottom_len)))

                if output[0] > 0.5:  # we use a
                    # tanh activation function so result will be between -1 and 1. if over 0.5 then jump
                    bird.jump()

                base.move()

                rem = []
                add_pipe = False
                for pipe in pipes:
                    pipe.move()
                    # check for collision
                    for bird in birds:
                        if pipe.colide(bird, win):
                            ge[birds.index(bird)].fitness -= 1
                            nets.pop(birds.index(bird))
                            ge.pop(birds.index(bird))
                            birds.pop(birds.index(bird))

                    if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                        rem.append(pipe)

                    if not pipe.passed and pipe.x < bird.x:
                        pipe.passed = True
                        add_pipe = True

                    if add_pipe:
                        score += 1
                        # can add this line to give more reword for passing through a pipe
                        for genome in ge:
                            genome.fitness += 5

                        new_pipe = Pipe(700)
                        new_pipe.set_height()
                        pipes.append(new_pipe)  # random height pipes

                    for r in rem:
                        pipes.remove(r)

                    for bird in birds:
                        if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                            nets.pop(birds.index(bird))
                            ge.pop(birds.index(bird))
                            birds.pop(birds.index(bird))

                    draw_window(win, birds, pipes, base, score, pipe_indx)


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
