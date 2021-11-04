import pygame
from pygame.locals import *
import time
import random

SIZE = 40
BACKGROUND_COLOR = (92, 25, 84)

class Apple:
    def __init__(self, parent_screen):
        self.image = pygame.image.load("apple.png").convert()
        self.parent_screen = parent_screen
        self.x = SIZE
        self.y = SIZE

    def draw (self):

        self.parent_screen.blit(self.image, (self.x, self.y))
        pygame.display.flip()

    def move(self):
        self.x = random.randint(0,24)*SIZE
        self.y = random.randint(0,19)*SIZE

class Snake:
    def __init__(self, parent_screen, length):
        self.length = length
        self.parent_screen = parent_screen
        self.block = pygame.image.load("block4.png").convert()
        self.direction = 'down'
        self.x = [SIZE]*length
        self.y = [SIZE]*length

    def increase_length(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)




    #After moving command drawing function
    def draw (self):
        self.parent_screen.fill(BACKGROUND_COLOR)
        for i in range(self.length):
            self.parent_screen.blit(self.block, (self.x[i], self.y[i]))
        pygame.display.flip()
    #Changin direction functions
    def move_left(self):
        self.direction = 'left'
    def move_right(self):
        self.direction = 'right'
    def move_up(self):
        self.direction = 'up'
    def move_down(self):
        self.direction = 'down'
    #Moving according to direction
    def walk(self):

        for i in range(self.length-1, 0, -1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        if self.direction == 'up':
            self.y[0] -= SIZE
        if self.direction == 'down':
            self.y[0] += SIZE
        if self.direction == 'left':
            self.x[0] -= SIZE
        if self.direction == 'right':
            self.x[0] += SIZE
        self.draw()

class Game:

    def __init__(self):
        # Creating display screen
        self.surface = pygame.display.set_mode((1000, 800))
        self.surface.fill((92, 25, 84))
        self.snake = Snake(self.surface, 2)
        self.snake.draw()
        self.apple = Apple(self.surface)
        self.apple.draw()

    def is_collision(self, x1, y1, x2, y2):
        if x1 >= x2 and x1 < x2 + SIZE:
            if y1 >= y2 and y1 < y2 +SIZE:
                return True

        return False

    def display_score(self):
        pygame.font.init()
        font = pygame.font.SysFont('arial', 30)
        score = font.render(f"Score: {self.snake.length}", True, (255, 255, 255))
        self.surface.blit(score, (850, 10))

    def play(self):
        self.snake.walk()
        self.apple.draw()
        self.display_score()
        pygame.display.flip()

        #snake coliding with apple
        if self.is_collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.snake.increase_length()
            self.apple.move()

        #snake coliding with itself
        for i in range(1, self.snake.length):
            if self.is_collision(self.snake.x[0], self.snake.y[0], self.snake.x[i], self.snake.y[i]):
                raise Exception('Game Over!')


    def show_game_over(self):
        self.surface.fill(BACKGROUND_COLOR)
        font = pygame.font.SysFont('arial', 30)
        line1 = font.render(f"Game is over! Your Score: {self.snake.length}", True, (255, 255, 255))
        self.surface.blit(line1, (200, 300))
        line2 = font.render(f"To play again press Enter. To exit press Escape!", True, (255, 255, 255))
        self.surface.blit(line2, (200, 300))
        pygame.display.flip()


    def reset(self):
        self.snake = Snake(self.surface, 2)
        self.apple = Apple(self.surface)

    def run(self):
        gaming = True
        pause = False
        # Run the game except quit event and movement commands
        while gaming:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.type == K_ESCAPE:
                        gaming = False

                    if event.key == K_RETURN:
                        pause = False

                    if event.key == K_UP:
                        self.snake.move_up()
                    if event.key == K_DOWN:
                        self.snake.move_down()
                    if event.key == K_RIGHT:
                        self.snake.move_right()
                    if event.key == K_LEFT:
                        self.snake.move_left()

                elif event.type == QUIT:
                    gaming = False

            try:
                if not pause:
                    self.play()
            except Exception as e:
                self.show_game_over()
                pause = True
                self.reset()
            time.sleep(0.2)




if __name__ == "__main__" :
    game = Game()
    game.run()



