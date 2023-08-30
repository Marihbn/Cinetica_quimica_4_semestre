import pygame
import numpy as np

# Cores do fundo e partículas
preto = (0, 0, 0)
azul = (0, 0, 255)
vermelho = (255, 0, 0)

def distancia(p1, p2):
    return p2.posicao - p1.posicao

def distancia_euclidiana(p1, p2):
    return np.linalg.norm(p2.posicao - p1.posicao)

def velocidade_relativa(p1, p2):
    return p2.velocidade - p1.velocidade

def escolhe_indice(matriz):
    # Find indices where the element is 0
    zero_indices_i, zero_indices_j = np.where(matriz == 0)
    
    # Check if there are any 0 elements in the array
    if len(zero_indices_i) > 0:
        # Choose a random index from the zero_indices
        u = np.random.randint(0, len(zero_indices_i))
        return (zero_indices_i[u], zero_indices_j[u])
    else:
        return False

def colisao_particulas(p1, p2):
    d = distancia_euclidiana(p1, p2)
    if d <= (p1.raio + p2.raio):
        if np.dot(velocidade_relativa(p1, p2), distancia(p1, p2)) < 0:
            dx = p2.posicao[0] - p1.posicao[0]
            dy = p2.posicao[1] - p1.posicao[1]
            M = np.array([[dx/d, dy/d], [-dy/d, dx/d]])
            V1_rt = M @ p1.velocidade
            V2_rt = M @ p2.velocidade
            alfa = p1.massa/p2.massa
            beta = p2.massa/p1.massa
            V1_rt[0], V2_rt[0] = ((1-alfa)/(1+alfa))*V1_rt[0] + (2/(1+alfa))*V2_rt[0], ((1-beta)/(1+beta))*V2_rt[0] + (2/(1+beta))*V1_rt[0]
            p1.velocidade = np.linalg.solve(M, V1_rt)
            p2.velocidade = np.linalg.solve(M, V2_rt)
            return


# Classe para representar as partículas
class Particula:
    def __init__(self, x, y, vx, vy, r, c):
        self.posicao = np.array([x, y])
        self.raio = r
        self.cor = c
        self.velocidade = np.array([vx, vy])
        self.massa = r**2

    def colisao_parede(self, sistema):
        if self.posicao[0] - self.raio < 0 or self.posicao[0] + self.raio > sistema.largura:
            self.velocidade[0] *= -1

        if self.posicao[1] - self.raio < 0 or self.posicao[1] + self.raio > sistema.altura:
            self.velocidade[1] *= -1

    def __add__(self, dt):
        self.posicao = self.posicao + self.velocidade*dt
        return

    def desenhar(self, superficie):
        pygame.draw.circle(superficie, self.cor, (self.posicao[0], self.posicao[1]), self.raio)

class Sistema:
    def __init__(self, U, V, N1, N2, r1, r2):
        self.energia_interna = U
        self.volume = V
        self.numero_de_particulas_1 = N1
        self.numero_de_particulas_2 = N2
        self.numero_de_particulas = N1 + N2
        self.raio_particula_1 = r1
        self.raio_particula_2 = r2
        self.largura = int(np.sqrt(self.volume))
        self.altura = int(np.sqrt(self.volume))

    def main(self, dt):
        pygame.init()
        window = pygame.display.set_mode((self.largura, self.altura))
        pygame.display.set_caption("Simulação de Colisão de Partículas")

        particulas = []

        posicoes_ocupadas_1 = np.zeros((self.altura, self.largura))
        posicoes_ocupadas_1[0: self.raio_particula_1, :] = 1
        posicoes_ocupadas_1[self.altura - self.raio_particula_1: self.altura, :] = 1
        posicoes_ocupadas_1[:, 0:self.raio_particula_1] = 1
        posicoes_ocupadas_1[:, self.largura - self.raio_particula_1: self.altura] = 1
        posicoes_ocupadas_2 = np.zeros((self.altura, self.largura))
        posicoes_ocupadas_2[0: self.raio_particula_2, :] = 1
        posicoes_ocupadas_2[self.altura - self.raio_particula_2: self.altura, :] = 1
        posicoes_ocupadas_2[:, 0:self.raio_particula_2] = 1
        posicoes_ocupadas_2[:, self.largura - self.raio_particula_2: self.altura] = 1
        
        for _ in range(self.numero_de_particulas_1):
            x_geracao, y_geracao = escolhe_indice(posicoes_ocupadas_1)
            posicoes_ocupadas_1[(x_geracao-2*self.raio_particula_1):(x_geracao+2*self.raio_particula_1), (y_geracao-2*self.raio_particula_1):(y_geracao+2*self.raio_particula_1)] = 1
            posicoes_ocupadas_2[(x_geracao-(self.raio_particula_1 + self.raio_particula_2)):(x_geracao+(self.raio_particula_1 + self.raio_particula_2)), (y_geracao-(self.raio_particula_1 + self.raio_particula_2)):(y_geracao+(self.raio_particula_1 + self.raio_particula_2))] = 1
            particula = Particula(x_geracao, y_geracao, 5, 5, self.raio_particula_1, 'blue')
            particulas.append(particula)

        for _ in range(self.numero_de_particulas_2):
            x_geracao, y_geracao = escolhe_indice(posicoes_ocupadas_2)
            posicoes_ocupadas_2[(x_geracao-2*self.raio_particula_2):(x_geracao+2*self.raio_particula_2), (y_geracao-2*self.raio_particula_2):(y_geracao+2*self.raio_particula_2)] = 1
            particula = Particula(x_geracao, y_geracao, 5, 5, self.raio_particula_1, 'red')
            particulas.append(particula)


        simulating = True
        clock = pygame.time.Clock()

        while simulating:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    simulating = False

            window.fill(preto)

            for particula in particulas:
                particula + dt
                particula.desenhar(window)
                particula.colisao_parede(self)

            for i in range(self.numero_de_particulas):
                for j in range(i + 1, self.numero_de_particulas):
                    colisao_particulas(particulas[i], particulas[j])

            pygame.display.update()
            clock.tick(10000/(dt**2))

        pygame.quit()

Sistema_particulas = Sistema(100, 360000, 10, 10, 15, 10)

if __name__ == "__main__":
    Sistema_particulas.main(0.01)
