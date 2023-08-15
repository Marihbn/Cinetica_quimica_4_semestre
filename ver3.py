import pygame
import numpy as np

pygame.init()

# Simulação da caixa por meio de uma janela
width, height = 600, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simulação de Colisão de Partículas")

# Cores do fundo e partículas
black = (0, 0, 0)
blue = (0, 0, 255)

# Classe para representar as partículas
class Particle:
    def __init__(self, x, y, radius):
        self.position = np.array([x, y])
        self.radius = radius
        self.color = blue
        self.velocity = np.array([10, 10])
        self.mass = radius**2

    def colisao_parede(self):
        if (self.position[0] - self.radius <= 0) or (self.position[0] + self.radius >= width):
            self.velocity[0] *= -1
        if (self.position[1] - self.radius <= 0) or (self.position[1] + self.radius >= height):
            self.velocity[1] *= -1

    def move(self):
        self.position = (self.position + self.velocity).astype(int)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.position[0]), int(self.position[1])), self.radius)

def distance(p1, p2):
    return np.sqrt((p1.position[0] - p2.position[0])**2 + (p1.position[1] - p2.position[1])**2)

def relative_speed(p1, p2):
    return np.sqrt((p1.velocity[0] - p2.velocity[0])**2 + (p1.velocity[1] - p2.velocity[1])**2)

def collide_particles(p1, p2):
    lamb = p1.mass / p2.mass
    beta = p2.mass / p1.mass
    if distance(p1, p2) <= (p1.radius + p2.radius):
        if relative_speed(p1, p2) > 0:
            new_velocity_1 = (2 / (1 + lamb)) * p2.velocity - ((1 - lamb) / (1 + lamb)) * p1.velocity
            new_velocity_2 = (2 / (1 + beta)) * p1.velocity - ((1 - beta) / (1 + beta)) * p2.velocity
            p1.velocity, p2.velocity = new_velocity_1, new_velocity_2

def main():
    pygame.init()
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Simulação de Colisão de Partículas")

    particles = []
    num_particles = 100

    #for _ in range(num_particles):
     #   particle = Particle(np.random.randint(0, width), np.random.randint(0, height), 50)
      #  particles.append(particle)

    #for _ in range(num_particles):
     #   radius = np.random.randint(5, 20)  # Defina o intervalo de tamanhos desejados
      #  particle = Particle(np.random.randint(0, width), np.random.randint(0, height), radius)
       # particles.append(particle)

    sizes = [5, 10]  # Lista de tamanhos possíveis
    for _ in range(num_particles):
        radius = np.random.choice(sizes)  # Escolhe um tamanho da lista
        particle = Particle(np.random.randint(0, width), np.random.randint(0, height), radius)
        particles.append(particle)

    #sizes_and_colors = [
    #(5, (0, 0, 255)),   # Tamanho 10, cor azul
    #(10, (255, 0, 0)),   # Tamanho 15, cor vermelha
    #(20, (0, 255, 0)),   # Tamanho 20, cor verde
    # Adicione mais pares de tamanho e cor conforme necessário
#]



    simulating = True
    clock = pygame.time.Clock()

    while simulating:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulating = False

        window.fill(black)

        for particle in particles:
            particle.move()
            particle.draw(window)
            particle.colisao_parede()

        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                collide_particles(particles[i], particles[j])

        pygame.display.update()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()