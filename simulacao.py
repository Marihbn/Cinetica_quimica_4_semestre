#importações
import matplotlib.colors as mcolors
import tkinter as tk
from vpython import *



win = 500

Natoms = 100  # número de atomos

L = 1 # cubo com lado de comprimento L
gray =  vector(190/255, 190/255, 190/255) # cor das bordas
mass = 4E-3/6E23 # massa do hélio
Ratom = 0.03 # tamanho exagerado do átomo
k = 1.4E-23 # constante de boltzman
T = 300 # temperatura ambiente
dt = 1E-5

animation = tk.Canvas(width=win, height=win)
animation.range = L
animation.title = 'Um gás de "esferas rígidas"'
s = """  Distribuições teóricas e médias de velocidade (metros/segundo).
Inicialmente, todos os átomos têm a mesma velocidade, mas as colisões
alteram as velocidades dos átomos que colidem. Um dos átomos é
marcado e deixa um rastro para que seja possível acompanhar seu trajeto.
  
"""
animation.caption = s

d = L/2+Ratom
r = 0.005
boxbottom = curve(color=gray, radius=r)
boxbottom.append([vector(-d,-d,-d), vector(-d,-d,d), vector(d,-d,d), vector(d,-d,-d), vector(-d,-d,-d)])
boxtop = curve(color=gray, radius=r)
boxtop.append([vector(-d,d,-d), vector(-d,d,d), vector(d,d,d), vector(d,d,-d), vector(-d,d,-d)])
vert1 = curve(color=gray, radius=r)
vert2 = curve(color=gray, radius=r)
vert3 = curve(color=gray, radius=r)
vert4 = curve(color=gray, radius=r)
vert1.append([vector(-d,-d,-d), vector(-d,d,-d)])
vert2.append([vector(-d,-d,d), vector(-d,d,d)])
vert3.append([vector(d,-d,d), vector(d,d,d)])
vert4.append([vector(d,-d,-d), vector(d,d,-d)])

Atoms = []
p = []
apos = []
pavg = sqrt(2*mass*1.5*k*T) # energia cinética média p**2/(2massa) = (3/2)kT
    
for i in range(Natoms):
    x = L*random()-L/2
    y = L*random()-L/2
    z = L*random()-L/2
    if i == 0:
        Atoms.append(sphere(pos=vector(x,y,z), radius=Ratom, color=color.cyan, make_trail=True, retain=100, trail_radius=0.3*Ratom))
    else: Atoms.append(sphere(pos=vector(x,y,z), radius=Ratom, color=gray))
    apos.append(vec(x,y,z))
    theta = pi*random()
    phi = 2*pi*random()
    px = pavg*sin(theta)*cos(phi)
    py = pavg*sin(theta)*sin(phi)
    pz = pavg*cos(theta)
    p.append(vector(px,py,pz))

deltav = 100 # agrupamento para o histograma de velocidades

def barx(v):
    return int(v/deltav) # Índice no array de barras

nhisto = int(4500/deltav)
histo = []
for i in range(nhisto): histo.append(0.0)
histo[barx(pavg/mass)] = Natoms

gg = graph( width=win, height=0.4*win, xmax=3000, align='left',
    xtitle='Velocidade, m/s', ytitle='Número de átomos', ymax=Natoms*deltav/1000)

theory = gcurve( color=color.blue, width=2 )
dv = 10
for v in range(0,3001+dv,dv):  # predição teórica
    theory.plot( v, (deltav/dv)*Natoms*4*pi*((mass/(2*pi*k*T))**1.5) *exp(-0.5*mass*(v**2)/(k*T))*(v**2)*dv )

accum = []
for i in range(int(3000/deltav)): accum.append([deltav*(i+.5),0])
vdist = gvbars(color=color.red, delta=deltav )

def interchange(v1, v2):  # remover da barra v1, adicionar à barra v2
    barx1 = barx(v1)
    barx2 = barx(v2)
    if barx1 == barx2:  return
    if barx1 >= len(histo) or barx2 >= len(histo): return
    histo[barx1] -= 1
    histo[barx2] += 1
    
def checkCollisions():
    hitlist = []
    r2 = 2*Ratom
    r2 *= r2
    for i in range(Natoms):
        ai = apos[i]
        for j in range(i) :
            aj = apos[j]
            dr = ai - aj
            if mag2(dr) < r2: hitlist.append([i,j])
    return hitlist

nhisto = 0 # número de snapshots do histograma para fazer a média

while True:
    rate(300)
    # acumular e fazer a média dos snapshots do histograma
    for i in range(len(accum)): accum[i][1] = (nhisto*accum[i][1] + histo[i])/(nhisto+1)
    if nhisto % 10 == 0:
        vdist.data = accum
    nhisto += 1

    # atualiza todas as posições
    for i in range(Natoms): Atoms[i].pos = apos[i] = apos[i] + (p[i]/mass)*dt
    
    # verifica colisões
    hitlist = checkCollisions()

    # se ocorrerem colisões, atualizar os momentos dos dois átomos
    for ij in hitlist:
        i = ij[0]
        j = ij[1]
        ptot = p[i]+p[j]
        posi = apos[i]
        posj = apos[j]
        vi = p[i]/mass
        vj = p[j]/mass
        vrel = vj-vi
        a = vrel.mag2
        if a == 0: continue;  # velocidades exatamente iguais
        rrel = posi-posj
        if rrel.mag > Ratom: continue # se um átomo atravessou completamente o outro
    
        # theta é o ângulo entre a velocidade relativa e o deslocamento relativo
        dx = dot(rrel, vrel.hat)       # rrel.mag*cos(theta)
        dy = cross(rrel, vrel.hat).mag # rrel.mag*sin(theta)
        # alpha é o ângulo do triângulo formado por rrel, o caminho do átomo j e uma linha
        # do centro do átomo i ao centro do átomo j, onde o átomo j atinge o átomo i:
        alpha = asin(dy/(2*Ratom)) 
        d = (2*Ratom)*cos(alpha)-dx # distância percorrida dentro do átomo a partir do primeiro contato
        deltat = d/vrel.mag         # tempo gasto se movendo do primeiro contato até a posição dentro do átomo
        
        posi = posi-vi*deltat # voltar para a configuração de contato
        posj = posj-vj*deltat
        mtot = 2*mass
        pcmi = p[i]-ptot*mass/mtot # transformar os momentos para o referencial do centro de massa (CM)
        pcmj = p[j]-ptot*mass/mtot
        rrel = norm(rrel)
        pcmi = pcmi-2*pcmi.dot(rrel)*rrel # ricochetear no referencial do centro de massa (CM)
        pcmj = pcmj-2*pcmj.dot(rrel)*rrel
        p[i] = pcmi+ptot*mass/mtot # transformar os momentos de volta para o referencial laboratorial
        p[j] = pcmj+ptot*mass/mtot
        apos[i] = posi+(p[i]/mass)*deltat # avance deltat no tempo
        apos[j] = posj+(p[j]/mass)*deltat
        interchange(vi.mag, p[i].mag/mass)
        interchange(vj.mag, p[j].mag/mass)
    
    for i in range(Natoms):
        loc = apos[i]
        if abs(loc.x) > L/2:
            if loc.x < 0: p[i].x =  abs(p[i].x)
            else: p[i].x =  -abs(p[i].x)
        
        if abs(loc.y) > L/2:
            if loc.y < 0: p[i].y = abs(p[i].y)
            else: p[i].y =  -abs(p[i].y)
        
        if abs(loc.z) > L/2:
            if loc.z < 0: p[i].z =  abs(p[i].z)
            else: p[i].z =  -abs(p[i].z)
    
