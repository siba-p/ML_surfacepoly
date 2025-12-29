import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from tqdm import tqdm
import seaborn as sns

N = 20

xpatchlevel = 2
ypatchlevel = 2

fracA = 0.1
fracB = 1-fracA

kB = 8.314
T = 1
ns = 1.0 #0.4 # J/kB*T

tsteps = 1000

prob = [fracA, fracB]

counter = 1

lattice = np.zeros((N,N))

def genOrderPatch():
    for i in range(N):
        for j in range(N):
            f1 = int(i/(N/xpatchlevel))%2
            f2 = int(j/(N/ypatchlevel))%2
            if((f1+f2)%2 == 0):
                lattice[i,j] = -1
            else:
                lattice[i,j] = +1

def genRandomPatch():
    for i in range(N):
        for j in range(N):
            lattice[i,j] = np.random.choice([+1, -1], p = prob)
            
def genRandomPatch2():
    
    global lattice
    
    lattice = -1*np.ones((N,N))
    
    sel = np.random.choice(N*N, size = int(np.round(fracA*N*N)), replace=False)
    
    for a in range(len(sel)):
        i = int(sel[a]/N)
        j = int(sel[a]%N)
        lattice[i, j] = +1


def Neumann(pos):
    x = pos[0]
    y = pos[1]
    neighb = np.zeros(4)
    neighb[0] = lattice[(x+1)%N,y]
    neighb[1] = lattice[(x-1)%N,y]
    neighb[2] = lattice[x,(y+1)%N]
    neighb[3] = lattice[x,(y-1)%N]
    return neighb

def MooNotNeu(pos):
    x = pos[0]
    y = pos[1]
    neighb = np.zeros(4)
    neighb[0] = lattice[(x+1)%N,(y+1)%N]
    neighb[1] = lattice[(x+1)%N,(y-1)%N]
    neighb[2] = lattice[(x-1)%N,(y+1)%N]
    neighb[3] = lattice[(x-1)%N,(y-1)%N]
    return neighb    
    
def Moore(pos):
    x = pos[0]
    y = pos[1]
    neighb = np.zeros(8)
    neighb[0] = lattice[(x+1)%N,y]
    neighb[1] = lattice[(x-1)%N,y]
    neighb[2] = lattice[x,(y+1)%N]
    neighb[3] = lattice[x,(y-1)%N]
    neighb[4] = lattice[(x+1)%N,(y+1)%N]
    neighb[5] = lattice[(x+1)%N,(y-1)%N]
    neighb[6] = lattice[(x-1)%N,(y+1)%N]
    neighb[7] = lattice[(x-1)%N,(y-1)%N]
    return neighb

def isSpinFlip(pos):
    x = pos[0]
    y = pos[1]
    mod = 1000*(fracA*N*N - (1-fracA)*N*N)/(N*N)
    dS = -2*lattice[x,y]
    nb = Moore(pos)
    dN = np.sum(nb)
    dE = -1*dN*dS
    if(dE <= -1*4*mod/(N*N)):
        return True
    elif(np.random.random() < np.exp(-ns*dE)):
        return True
    else:
        return False

def init():
    global im
    im = plt.imshow(lattice)

def sweepIsing():
    global counter
    global lattice
    temp = np.copy(lattice)
    #print(counter)
    counter = counter + 1
    for i in range(N):
        for j in range(N):
            if isSpinFlip([i,j]):
                temp[i,j] = temp[i,j]*-1
            else:
                temp[i,j] = temp[i,j]
    lattice = np.copy(temp)
    del(temp)
    
def MCIsing():
    global counter
    global lattice
    temp = np.copy(lattice)
    #print(counter)
    counter = counter + 1
    space = range(N)
    i, j = np.random.choice(space, 2, replace = True)
    if isSpinFlip([i,j]):
        lattice[i,j] = lattice[i,j]*-1

def runIsing():
    for t in tqdm(range(tsteps)):
        MCIsing()

def animate(i):
    for i in range(10):
        MCIsing()
    im.set_array(lattice)

def genSurfaces(num):
    
    global lattice

    genRandomPatch2()

    needfrac= fracA#0.2

    #frame_list = []
    #master_list = []

    temp_list = []

    delt = 1000000

    for t in tqdm(range(delt)):
        MCIsing()
        deg = np.sum(lattice)
        tfracA = ((deg/(N*N)) + 1)/2
        temp_list.append(tfracA)
        if(t > 1000 and np.round(tfracA,4) == np.round(needfrac,4)):
            np.save("surface_"+str(num+1)+".npy", lattice)
            return 1, temp_list
        #    break
    return 0, temp_list
#plt.plot(temp_list)    
def fraction(lattice):
    countA = 0
    countB = 0    
    for i in lattice:
        for j in i:        
            if j == -1:
                countA += 1
            else:
                countB += 1
    f = float(countA/(countA+countB))       
    return(f)


pframes = 5
fig, axes = plt.subplots(pframes, 1, figsize=(15, 10))
ctr = 0
#for i in range(pframes, pframes+1):
while(ctr < pframes):
    print("Surface "+str(ctr+1))
    state, tlist = genSurfaces(ctr)
    
    if(state == 1):
        print("Surface generation successful!")
#        sns.heatmap(lattice, annot=True, cbar=False, cmap='coolwarm', linewidths=0.5, ax=axes[ctr])
       # print(np.shape(lattice))
#        axes[ctr].set_title(f'Surface {ctr + 1} - Fraction: {fraction(lattice):.2f}')
        ctr += 1
    else:
        print("Surface generation failed! Trying again")
#plt.tight_layout()
#plt.show()
print(fraction(lattice))
