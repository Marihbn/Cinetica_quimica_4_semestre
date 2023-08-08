import pygame
import math
import numpy as np

pygame.init()

# Simulação da caixa por meio de uma janela
width, height = 600, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simulação de Colisão de Partículas")

# Cores do fundo e particulas
black = (0, 0, 0)
blue = (0, 0, 255)

# Classe para representar as partículas
class Particle:
    def __init__(self, x, y, raio):
        self.posicao = np.array([x, y])
        self.raio = raio
        self.cor = blue
        self.velocidade = np.array([5, 6])
        self.massa = raio**2

    def colisao_parede(self):
        if (self.posicao[0] - self.raio <= 0) or (self.posicao[0] + self.raio >= width):
            self.velocidade[0] *= -1
        if (self.posicao[1] - self.raio <= 0) or (self.posicao[1] + self.raio >= height):
            self.velocidade[1] *= -1
        
    #def distancia_particulas(self, outra_particula):
    #    return np.sqrt((self.posicao[0] - outra_particula.posicao[0])**2 + (self.posicao[1] - outra_particula.posicao[1])**2)

    def colisao_particulas(self, outra_particula):
        lamb = self.massa/outra_particula.massa
        beta = outra_particula.massa/self.massa
        vel1_init = self.velocidade.copy()
        vel2_init = outra_particula.velocidade.copy()
        if np.sqrt((self.posicao[0] - outra_particula.posicao[0])**2 + (self.posicao[1] - outra_particula.posicao[1])**2) <= (self.raio + outra_particula.raio):
            self.velocidade = (2/(1+lamb))*vel2_init - ((1-lamb)/(1+lamb))*vel1_init
            outra_particula.velocidade = (2/(1+beta))*vel1_init - ((1-beta)/(1+beta))*vel2_init

    def movimento(self):
        self.posicao += self.velocidade

    def desenho(self):
        pygame.draw.circle(window, self.cor, (int(self.posicao[0]), int(self.posicao[1])), self.raio)

# Criando as partículas
particula_1 = Particle(width // 3, height - 100 // 2, 10)
particula_2 = Particle(2 * width // 3, height // 2, 10)

# Rodando a simulação
sim = True
c = pygame.time.Clock()

while sim:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sim = False

    window.fill(black)

    # Movendo e desenhando as partículas
    particula_1.movimento()
    particula_2.movimento()

    particula_1.desenho()
    particula_2.desenho()

    # Detectando colisão
    particula_1.colisao_parede()
    particula_2.colisao_parede()

    particula_1.colisao_particulas(particula_2)

    pygame.display.update()
    c.tick(60)

pygame.quit()
