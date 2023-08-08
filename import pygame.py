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

    def movimento(self):
        self.posicao = (self.posicao + self.velocidade).astype(int)

    def desenho(self):
        pygame.draw.circle(window, self.cor, (int(self.posicao[0]), int(self.posicao[1])), self.raio)

def distancia_particulas(p1, p2):
    return np.sqrt((p1.posicao[0] - p2.posicao[0])**2 + (p1.posicao[1] - p2.posicao[1])**2)

def velocidade_relativa(p1, p2):
    return np.sqrt((p1.velocidade[0] - p2.velocidade[0])**2 + (p1.velocidade[1] - p2.velocidade[1])**2)

def colisao_particulas(p1, p2):
    lamb = p1.massa/p2.massa
    beta = p2.massa/p1.massa
    if distancia_particulas(p1, p2) <= (p1.raio + p2.raio):
        if velocidade_relativa(p1, p2) > 0:
            p1.velocidade, p2.velocidade = (2/(1+lamb))*p2.velocidade - ((1-lamb)/(1+lamb))*p1.velocidade, (2/(1+beta))*p1.velocidade - ((1-beta)/(1+beta))*p2.velocidade
            #p1.velocidade, p2.velocidade = p1.velocidade.astype(np.int32), p2.velocidade.astype(np.int32)

# Criando as partículas
particula_1 = Particle(width // 3, height - 100 // 2, 80)
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

    colisao_particulas(particula_1, particula_2)

    pygame.display.update()
    c.tick(60)

pygame.quit()
