import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from tqdm import tqdm
import seaborn as sns

N1 = 8
N2 = 5

xpatchlevel = 2
ypatchlevel = 2

fracA = 1.0
fracB = 1-fracA

kB = 8.314
T = 1
ns = 0.4 #0.4 # J/kB*T

tsteps = 100000

prob = [fracA, fracB]

counter = 1

lattice = np.zeros((N1,N2))
def genOrderPatch():
    for i in range(N1):
        for j in range(N2):
            f1 = int(i/(N1/xpatchlevel))%2
            f2 = int(j/(N2/ypatchlevel))%2
            if((f1+f2)%2 == 0):
                lattice[i,j] = -1
            else:
                lattice[i,j] = +1
def genRandomPatch():
    for i in range(N1):
        for j in range(N2):
            lattice[i,j] = np.random.choice([+1, -1], p = prob)

def genRandomPatch2():
    
    global lattice
    
    lattice = -1*np.ones((N1,N2))
    
    sel = np.random.choice(range(N1*N2), size = int(np.round(fracA*N1*N2)), replace=False)
    
    for a in range(len(sel)):
        i = int(sel[a]%N1)
        j = int(sel[a]/N1)
        lattice[i, j] = +1

def Neumann(pos):
    x = pos[0]
    y = pos[1]
    neighb = np.zeros(4)
    neighb[0] = lattice[(x+1)%N1,y]
    neighb[1] = lattice[(x-1)%N1,y]
    neighb[2] = lattice[x,(y+1)%N2]
    neighb[3] = lattice[x,(y-1)%N2]
    return neighb

def MooNotNeu(pos):
    x = pos[0]
    y = pos[1]
    neighb = np.zeros(4)
    neighb[0] = lattice[(x+1)%N1,(y+1)%N2]
    neighb[1] = lattice[(x+1)%N1,(y-1)%N2]
    neighb[2] = lattice[(x-1)%N1,(y+1)%N2]
    neighb[3] = lattice[(x-1)%N1,(y-1)%N2]
    return neighb    
    
def Moore(pos):
    x = pos[0]
    y = pos[1]
    neighb = np.zeros(8)
    neighb[0] = lattice[(x+1)%N1,y]
    neighb[1] = lattice[(x-1)%N1,y]
    neighb[2] = lattice[x,(y+1)%N2]
    neighb[3] = lattice[x,(y-1)%N2]
    neighb[4] = lattice[(x+1)%N1,(y+1)%N2]
    neighb[5] = lattice[(x+1)%N1,(y-1)%N2]
    neighb[6] = lattice[(x-1)%N1,(y+1)%N2]
    neighb[7] = lattice[(x-1)%N1,(y-1)%N2]
    return neighb

def isSpinFlip(pos):
    x = pos[0]
    y = pos[1]
    mod = 1000*(fracA*N1*N2 - (1-fracA)*N1*N2)/(N1*N2)
    dS = -2*lattice[x,y]
    nb = Neumann(pos)
    dN = np.sum(nb)
    dE = -1*dN*dS
#    if(dE <= -1*4*mod/(N1*N2)):
    if(dE <= 0.0):
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
    for i in range(N1):
        for j in range(N2):
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
    space1 = range(N1)
    i = np.random.choice(space1, 1, replace = True)
    space2 = range(N2)
    j = np.random.choice(space2, 1, replace = True)
    if isSpinFlip([i,j]):
        lattice[i,j] = lattice[i,j]*-1

def runIsing():
    for t in tqdm(range(tsteps)):
        MCIsing()

def animate(i):
    for i in range(10):
        MCIsing()
    im.set_array(lattice)

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
    print(countA, countB)
    return(f)
###################################
#####for surface only########
##############################
###############################
#################################
###################################

def genSurfaces(num):
    
    global lattice

    genRandomPatch2()

    needfrac= fracA#0.2

    #frame_list = []
    #master_list = []

    temp_list = []

    delt = tsteps

    for t in tqdm(range(delt)):
        MCIsing()
        deg = np.sum(lattice)
        tfracA = (-1*(deg/(N1*N2)) + 1)/2
        temp_list.append(tfracA)
        if(t > 10000 and np.round(tfracA,4) == np.round(needfrac,4)):
            np.save("surface_"+str(num+1)+".npy", lattice)
            return 1, temp_list
        #    break
    return 0, temp_list


#pframes = 10
#ctr = 0
#for i in range(pframes, pframes+1):
#while(ctr < pframes):
#    print("Surface "+str(ctr+1))
#    state, tlist = genSurfaces(ctr)
#    print(tlist)
#    plt.plot(tlist)
#    if(state == 1):
#        print("Surface generation successful!")
#        print(fraction(lattice))
#        ctr = ctr + 1
#    else:
#        print("Surface generation failed! Trying again")

########################
####Visulaise the surface
#########################

#fig, axs = plt.subplots(5, 2, figsize = (20,20))

#nSurf = 10

#for i in range(nSurf):
#    j = i%5
#    k = int(i/5)
#    temp = np.load("surface_"+str(i+1)+".npy")
#    axs[j, k].imshow(temp)
#plt.savefig("surface.png")
#plt.show()

###################################
#####for polymer only########
##############################
###############################
#################################
###################################
def genPolymers(num):
    
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
        tfracA = (-1*(deg/(N1*N2)) + 1)/2
        temp_list.append(tfracA)
        if(t > 1000 and np.round(tfracA,4) == np.round(needfrac,4)):
            np.save("polymer_"+str(num+1)+".npy", lattice.flatten())
            return 1, temp_list
        #    break
    return 0, temp_list
pframes = 100
ctr = 90
for i in range(pframes, pframes+1):
 while(ctr < pframes):
    print("Polymer "+str(ctr+1))
    state, tlist = genPolymers(ctr)
    plt.plot(tlist)
    if(state == 1):
        print("Polymer generation successful!")
        ctr = ctr + 1
    else:
        print("Polymer generation failed! Trying again")

#Visualisation polymer

fig, axs = plt.subplots(5, 2, figsize = (10,5))

#nSurf = 10

#for i in range(10,20):
#    j = i%5
#    k = int(i/5)
#    temp = np.load("polymer_"+str(i+1)+".npy")
##    print(np.sum(temp))
#    axs[j, k].imshow(np.expand_dims(temp, axis=0))
#plt.savefig(f"polymer_{fracA}.png")
#plt.show()
fig, axes = plt.subplots(5,2, figsize = (30,15))
n = 100
for i in range(90,n):
    data = np.load("polymer_"+str(i+1)+".npy")
    sns.heatmap(data.reshape(1,-1), ax=axes[(i-90)//2,(i-90)%2],annot=True)
plt.savefig(f"polymer_{fracA}.png")
plt.show()
