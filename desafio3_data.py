import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def distancia(p1, p2):
    return p2.posicao - p1.posicao

def distancia_euclidiana(p1, p2):
    return np.linalg.norm(p2.posicao - p1.posicao)

def velocidade_relativa(p1, p2):
    return p2.velocidade - p1.velocidade
        
def desenhar_particulas(ax, particle_list):
    particle_number = len(particle_list)
    circle = [None]*particle_number
    for i in range(particle_number):
        circle[i] = plt.Circle((particle_list[i].posicao[0], particle_list[i].posicao[1]), particle_list[i].raio, color=particle_list[i].cor, lw=1.5, zorder=20)
        ax.add_patch(circle[i])

def checar_colisao(p1, p2):
    d = distancia_euclidiana(p1, p2)
    if d <= (p1.raio + p2.raio):
        if np.dot(velocidade_relativa(p1, p2), distancia(p1, p2)) < 0:
            return True
    return False
        
def colisao_particulas(p1, p2):
    d = distancia_euclidiana(p1, p2)
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

def checar_reacao(p1, p2, eta):
    x_rand = np.random.choice(np.linspace(0, 1, 100))
    x_reac = x_rand + p1.reac + p2.reac
    if x_reac < eta:
        return True
    else:
        return False
        

def reacao_quimica(pA, pB):
    momento_total = pA.velocidade*pA.massa + pB.velocidade*pB.massa
    centro_de_massa = (pA.posicao*pA.massa + pB.posicao*pB.massa)/(pA.massa + pB.massa)
    massaC = pA.massa + pB.massa
    velocidadeC = momento_total/massaC
    raioC = np.sqrt(massaC)
    pC = Particula(centro_de_massa[0], centro_de_massa[1], velocidadeC[0], velocidadeC[1], raioC, 'green', 2)
    return pC

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
    
def conta_particulas(lista):
    num1, num2, num3 = 0, 0, 0
    for particula in lista:
        if particula.cor == 'red':
            num1 += 1
        if particula.cor == 'blue':
            num2 += 1
        if particula.cor == 'green':
            num3 += 1
            
    return num1, num2, num3
        
# Classe para representar as partículas
class Particula:
    def __init__(self, x, y, vx, vy, r, c, reatividade):
        self.posicao = np.array([x, y])
        self.raio = r
        self.cor = c
        self.velocidade = np.array([vx, vy])
        self.massa = r**2
        self.reac = reatividade

    def colisao_parede(self, sistema):
        if self.posicao[0] - self.raio < 0 or self.posicao[0] + self.raio > sistema.largura:
            self.velocidade[0] *= -1

        if self.posicao[1] - self.raio < 0 or self.posicao[1] + self.raio > sistema.altura:
            self.velocidade[1] *= -1

    def __add__(self, dt):
        self.posicao = self.posicao + self.velocidade*dt
        return

class Sistema:
    # U deverá ser grande o suficiente para velocidades médias > 1.
    # r1 e r2 deverão ser inteiros
    def __init__(self, U, V, N1, r1, reatv1, reatv2, eta):
        self.energia_interna = U
        self.volume = V
        self.numero_de_particulas_1 = N1
        self.numero_de_particulas_2 = 0
        self.numero_de_particulas_3 = 0
        self.numero_de_particulas = N1
        self.raio_particula_1 = r1
        self.largura = int(np.sqrt(V))
        self.altura = int(np.sqrt(V))
        self.particulas = []
        self.energia_interna_1 = U
        self.reatividade_1 = reatv1
        self.reatividade_2 = reatv2
        self.velocidade_2_media_1 = 2*self.energia_interna_1/(self.numero_de_particulas_1*(r1**2))
        self.velocidade_media_1 = np.sqrt(self.velocidade_2_media_1)
        self.lista_modulo_velocidades = []
        self.concentracoes = {'A': [N1],
                              'B': [0],
                              'C': [0]
                             }
        self.vel_relativas_A_B = []
        self.prop = 1/self.largura
        self.prob_reacao = eta
        
        posicoes_ocupadas_1 = np.zeros((self.altura, self.largura))
        posicoes_ocupadas_1[0: self.raio_particula_1, :] = 1
        posicoes_ocupadas_1[self.altura - self.raio_particula_1: self.altura, :] = 1
        posicoes_ocupadas_1[:, 0:self.raio_particula_1] = 1
        posicoes_ocupadas_1[:, self.largura - self.raio_particula_1: self.altura] = 1
        
        velocidades_1 = [np.random.randint(10, 50) for _ in range(self.numero_de_particulas_1)]
        vel_1_array = np.array(velocidades_1)
        vel_1_array = (vel_1_array/np.mean(vel_1_array))*self.velocidade_media_1
        
        for ind_vel in range(self.numero_de_particulas_1):
            # escolha da posição inicial
            x_geracao, y_geracao = escolhe_indice(posicoes_ocupadas_1)
            posicoes_ocupadas_1[(x_geracao-2*self.raio_particula_1):(x_geracao+2*self.raio_particula_1), (y_geracao-2*self.raio_particula_1):(y_geracao+2*self.raio_particula_1)] = 1
            
            # escolha da velocidade inicial
            vx_geracao = np.random.randint(-vel_1_array[ind_vel], vel_1_array[ind_vel])
            vy_geracao = np.sqrt(vel_1_array[ind_vel]**2 - vx_geracao**2)
            
            # criação da partícula
            particula = Particula(x_geracao, y_geracao, vx_geracao, vy_geracao, self.raio_particula_1, 'red', self.reatividade_1)
            self.particulas.append(particula)
    
    
    def main(self, dt, frame_period, n, N):
            
        #fig = plt.figure(figsize=(12, 12))
        #ax = fig.add_subplot(2,2,1)
#
        #hist = fig.add_subplot(2,2,2)
        #
        #conc = fig.add_subplot(2, 1, 2)
#
        #plt.subplots_adjust(bottom=0.2,left=0.15)
#
        #ax.axis('equal')
        #ax.axis([-1, 30, -1, 30])
        #
        #boxsize = self.altura
        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)
        #ax.set_xlim([0,boxsize])
        #ax.set_ylim([0,boxsize])
        #
        #desenhar_particulas(ax, self.particulas)
        #
        ## Graph Particles speed histogram
        #vel_mod = [np.linalg.norm(np.copy(self.particulas[i].velocidade)) for i in range(len(self.particulas))]
        #self.lista_modulo_velocidades += vel_mod
        #hist.hist(self.lista_modulo_velocidades, bins= 50, density = True, label = "Dados da Simulação")
        #hist.set_xlabel("Velocidade")
        #hist.set_ylabel("Densidade de Frequências")
        #
        #U_medio = self.energia_interna/len(self.particulas) 
        #m = self.raio_particula_1**2
        #v = np.arange(0.1, 4*(self.velocidade_media_1))
        #fv = m*np.exp(-m*v**2/(2*U_medio*(2/3)))/(2*np.pi*U_medio*(2/3))*2*np.pi*v
        #hist.set_ylim(0, 1.5*np.max(fv))
        #hist.plot(v,fv, label = "Distribuição de Maxwell–Boltzmann") 
        #hist.legend(loc ="upper right")
        #conc.plot(list(range(n+1)), self.concentracoes['A'], color = 'red', label = "H+")
        #conc.plot(list(range(n+1)), self.concentracoes['B'], color = 'blue', label = "H+ (cat)")
        #conc.plot(list(range(n+1)), self.concentracoes['C'], color = 'green', label = "H2")
        #conc.set_xlim(0, N)
        #conc.set_ylabel("Número de partículas")
        #conc.set_xlabel("Tempo")
        #conc.legend()
        #
        #plt.savefig(f'./GIF - Desafio3/Imagem {n}.jpg', 
        #        transparent = False,  
        #        facecolor = 'white'
        #       )
        #plt.close()
#
        for _ in range(frame_period):

            for particula in self.particulas:
                particula + dt
                particula.colisao_parede(self)
                if particula.posicao[0] - particula.raio < 0:
                    particulaB = Particula(particula.posicao[0], particula.posicao[1], particula.velocidade[0], particula.velocidade[1], self.raio_particula_1, 'blue', self.reatividade_2)
                    self.particulas.remove(particula)
                    self.particulas.append(particulaB)
            
            novo_particulas = self.particulas.copy()
            for i in range(self.numero_de_particulas):
                for j in range(i + 1, self.numero_de_particulas):
                    
                    #if (self.particulas[i].cor == 'red' and self.particulas[j].cor == 'blue') or (self.particulas[i].cor == 'blue' and self.particulas[j].cor == 'red'):
                    #    self.vel_relativas_A_B.append(np.linalg.norm(self.prop*(velocidade_relativa(self.particulas[i], self.particulas[j]))))
                    
                    if checar_colisao(self.particulas[i], self.particulas[j]):
                        
                        if checar_reacao(self.particulas[i], self.particulas[j], self.prob_reacao):
                            particulaC = reacao_quimica(self.particulas[i], self.particulas[j])
                            novo_particulas.remove(self.particulas[i]), novo_particulas.remove(self.particulas[j])
                            novo_particulas.append(particulaC)
                            
                        else:
                            colisao_particulas(self.particulas[i], self.particulas[j])
            
            self.particulas = novo_particulas
        
            numA, numB, numC = conta_particulas(self.particulas)
            self.numero_de_particulas = numA + numB + numC
            self.concentracoes['A'].append(numA)
            self.concentracoes['B'].append(numB)
            self.concentracoes['C'].append(numC)
            
            

        
        