import numpy as np
import matplotlib.pyplot as plt

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

def checar_reacao(p1, p2, v_min):
    if (p1.cor == 'red' and p2.cor == 'blue') or (p1.cor == 'blue' and p2.cor == 'red'):
        v_rel_r = -(np.dot(velocidade_relativa(p1, p2), distancia(p1, p2))/distancia_euclidiana(p1, p2))
        if v_rel_r > v_min:
            return True
    return False
        

def reacao_quimica(pA, pB):
    momento_total = pA.velocidade*pA.massa + pB.velocidade*pB.massa
    centro_de_massa = (pA.posicao*pA.massa + pB.posicao*pB.massa)/(pA.massa + pB.massa)
    massaC = pA.massa + pB.massa
    velocidadeC = momento_total/massaC
    raioC = int(np.sqrt(massaC))
    pC = Particula(centro_de_massa[0], centro_de_massa[1], velocidadeC[0], velocidadeC[1], raioC, 'green')
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

class Sistema:
    # U deverá ser grande o suficiente para velocidades médias > 1.
    # r1 e r2 deverão ser inteiros
    def __init__(self, U, V, N1, N2, r1, r2):
        self.energia_interna = U
        self.volume = V
        self.numero_de_particulas_1 = N1
        self.numero_de_particulas_2 = N2
        self.numero_de_particulas_3 = 0
        self.numero_de_particulas = N1 + N2
        self.raio_particula_1 = r1
        self.raio_particula_2 = r2
        self.largura = int(np.sqrt(V))
        self.altura = int(np.sqrt(V))
        self.particulas = []
        self.energia_interna_1 = U/(1 + (N2/N1)*(r2/r1)**2)
        self.energia_interna_2 = U - self.energia_interna_1
        self.velocidade_2_media_1 = 2*self.energia_interna_1/(self.numero_de_particulas_1*(r1**2))
        self.velocidade_2_media_2 = 2*self.energia_interna_2/(self.numero_de_particulas_2*(r2**2))
        self.velocidade_media_1 = np.sqrt(self.velocidade_2_media_1)
        self.velocidade_media_2 = np.sqrt(self.velocidade_2_media_2)
        self.lista_modulo_velocidades = []
        self.concentracoes = {'A': [N1],
                              'B': [N2],
                              'C': [0]
                             }
        
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
        
        velocidades_1 = [np.random.randint(10, 30) for _ in range(self.numero_de_particulas_1)]
        vel_1_array = np.array(velocidades_1)
        vel_1_array = (vel_1_array/np.mean(vel_1_array))*self.velocidade_media_1
        
        velocidades_2 = [np.random.randint(10, 30) for _ in range(self.numero_de_particulas_2)]
        vel_2_array = np.array(velocidades_2)
        vel_2_array = (vel_2_array/np.mean(vel_2_array))*self.velocidade_media_2
        
        for ind_vel in range(self.numero_de_particulas_1):
            # escolha da posição inicial
            x_geracao, y_geracao = escolhe_indice(posicoes_ocupadas_1)
            posicoes_ocupadas_1[(x_geracao-2*self.raio_particula_1):(x_geracao+2*self.raio_particula_1), (y_geracao-2*self.raio_particula_1):(y_geracao+2*self.raio_particula_1)] = 1
            posicoes_ocupadas_2[(x_geracao-(self.raio_particula_1 + self.raio_particula_2)):(x_geracao+(self.raio_particula_1 + self.raio_particula_2)), (y_geracao-(self.raio_particula_1 + self.raio_particula_2)):(y_geracao+(self.raio_particula_1 + self.raio_particula_2))] = 1
            
            # escolha da velocidade inicial
            vx_geracao = np.random.randint(-vel_1_array[ind_vel], vel_1_array[ind_vel])
            vy_geracao = np.sqrt(vel_1_array[ind_vel]**2 - vx_geracao**2)
            
            # criação da partícula
            particula = Particula(x_geracao, y_geracao, vx_geracao, vy_geracao, self.raio_particula_1, 'blue')
            self.particulas.append(particula)

        for ind_vel in range(self.numero_de_particulas_2):
            # escolha da posição inicial
            x_geracao, y_geracao = escolhe_indice(posicoes_ocupadas_2)
            posicoes_ocupadas_2[(x_geracao-2*self.raio_particula_2):(x_geracao+2*self.raio_particula_2), (y_geracao-2*self.raio_particula_2):(y_geracao+2*self.raio_particula_2)] = 1
            
            # escolha da velocidade inicial
            vx_geracao = np.random.randint(-vel_2_array[ind_vel], vel_2_array[ind_vel])
            vy_geracao = np.sqrt(vel_2_array[ind_vel]**2 - vx_geracao**2)
            
            # criação da partícula
            particula = Particula(x_geracao, y_geracao, vx_geracao, vy_geracao, self.raio_particula_2, 'red')
            self.particulas.append(particula)
    
    
    def main(self, dt, frame_period, n, N):
            
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(2,2,1)

        hist = fig.add_subplot(2,2,2)
        
        conc = fig.add_subplot(2, 1, 2)

        plt.subplots_adjust(bottom=0.2,left=0.15)

        ax.axis('equal')
        ax.axis([-1, 30, -1, 30])
        
        boxsize = self.altura
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim([0,boxsize])
        ax.set_ylim([0,boxsize])
        
        desenhar_particulas(ax, self.particulas)
        
        # Graph Particles speed histogram
        vel_mod = [np.linalg.norm(np.copy(self.particulas[i].velocidade)) for i in range(len(self.particulas))]
        self.lista_modulo_velocidades += vel_mod
        hist.hist(self.lista_modulo_velocidades, bins= 50, density = True, label = "Dados da Simulação")
        hist.set_xlabel("Velocidade")
        hist.set_ylabel("Densidade de Frequências")
        
        U_medio = self.energia_interna/len(self.particulas) 
        m = (self.particulas[0].massa + self.particulas[-1].massa)/2
        v = np.arange(0.1, 2*(self.velocidade_media_1 + self.velocidade_media_2))
        fv = m*np.exp(-m*v**2/(2*U_medio*(2/3)))/(2*np.pi*U_medio*(2/3))*2*np.pi*v
        hist.set_ylim(0, 1.5*np.max(fv))
        hist.plot(v,fv, label = "Distribuição de Maxwell–Boltzmann") 
        hist.legend(loc ="upper right")
        
        conc.plot(list(range(n+1)), self.concentracoes['A'], color = 'blue', label = "A")
        conc.plot(list(range(n+1)), self.concentracoes['B'], color = 'red', label = "B")
        conc.plot(list(range(n+1)), self.concentracoes['C'], color = 'green', label = "C")
        conc.set_xlim(0, N)
        conc.set_ylabel("Número de partículas")
        conc.set_xlabel("Tempo")
        conc.legend()
        
        plt.savefig(f'./GIF - Simulação Gases/Imagem {n}.jpg', 
                transparent = False,  
                facecolor = 'white'
               )
        plt.close()

        for _ in range(frame_period):

            for particula in self.particulas:
                particula + dt
                particula.colisao_parede(self)
            
            novo_particulas = self.particulas.copy()
            for i in range(self.numero_de_particulas):
                for j in range(i + 1, self.numero_de_particulas):
                    if checar_colisao(self.particulas[i], self.particulas[j]):
                        
                        if checar_reacao(self.particulas[i], self.particulas[j], self.velocidade_media_1*(1/3)):
                            self.numero_de_particulas_1 -= 1
                            self.numero_de_particulas_2 -= 1
                            self.numero_de_particulas_3 += 1
                            particulaC = reacao_quimica(self.particulas[i], self.particulas[j])
                            novo_particulas.append(particulaC)
                            novo_particulas.remove(self.particulas[i])
                            novo_particulas.remove(self.particulas[j])
                        else:
                            colisao_particulas(self.particulas[i], self.particulas[j])
            self.numero_de_particulas = len(novo_particulas)
            self.particulas = novo_particulas
            
        self.concentracoes['A'].append(self.numero_de_particulas_1)
        self.concentracoes['B'].append(self.numero_de_particulas_2)
        self.concentracoes['C'].append(self.numero_de_particulas_3)
        
        