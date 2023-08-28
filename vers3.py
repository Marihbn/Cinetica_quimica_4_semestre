import pygame
import numpy as np

# Cores do fundo e partículas
black = (0, 0, 0)
blue = (0, 0, 255)

width, height = 600, 600

# Classe para representar as partículas
class Particle:
    def __init__(self, x, y, radius):
        self.position = np.array([x, y])
        self.radius = radius
        self.color = blue
        self.velocity = np.array([1, 2])
        self.mass = radius**2

    def colisao_parede(self):
        if (self.position[0] - self.radius <= 0):
            self.velocity[0] = np.abs(self.velocity[0])
        if (self.position[0] + self.radius >= width):
            self.velocity[0] = -1*np.abs(self.velocity[0])
        if (self.position[1] - self.radius <= 0):
            self.velocity[1] = np.abs(self.velocity[1])
        if (self.position[1] + self.radius >= height):
            self.velocity[1] = -1*np.abs(self.velocity[1])

    def move(self, dt):
        self.position = (self.position + self.velocity*dt).astype(float)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (float(self.position[0]), float(self.position[1])), self.radius)

def distance(p1, p2):
    return np.sqrt((p1.position[0] - p2.position[0])**2 + (p1.position[1] - p2.position[1])**2)

def relative_speed(p1, p2):
    return p2.velocity - p1.velocity

def vector_distance(p1, p2):
    return p2.position - p1.position

def collide_particles(p1, p2):
    if distance(p1, p2) <= (p1.radius + p2.radius):
        new_velocity_1 = p1.velocity - (((2*p2.mass)/(p1.mass + p2.mass))*np.dot((p1.velocity - p2.velocity), (p1.position - p2.position))/((p1.radius + p2.radius)**2))*(p1.position - p2.position)
        new_velocity_2 = p2.velocity - (((2*p1.mass)/(p2.mass + p1.mass))*np.dot((p2.velocity - p1.velocity), (p2.position - p1.position))/((p2.radius + p1.radius)**2))*(p2.position - p1.position)
        p1.velocity, p2.velocity = new_velocity_1, new_velocity_2

def main():
    pygame.init()
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Simulação de Colisão de Partículas")

    #radius1 = 20
    #radius2 = 20
    #particle1 = Particle(2*radius1, (height/2) + (radius1/2), 5, 0, radius1)
    #particle2 = Particle(width - 2*radius2, (height/2) - (radius2/2), -5, 0, radius2)
    #particles = [particle1, particle2]
    particles = []
    num_particles = 50

    sizes = [8, 10]  # Lista de tamanhos possíveis
    for _ in range(num_particles):
        radius = np.random.choice(sizes)  # Escolhe um tamanho da lista
        particle = Particle(np.random.randint(0, width), np.random.randint(0, height), radius)
        particles.append(particle)


    simulating = True
    clock = pygame.time.Clock()

    while simulating:
        dt = clock.get_time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulating = False

        window.fill(black)

        for particle in particles:
            particle.move(dt)
            particle.draw(window)
            particle.colisao_parede()

        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                collide_particles(particles[i], particles[j])

        pygame.display.update()
        clock.tick(10000)

    pygame.quit()

if __name__ == "__main__":
    main()