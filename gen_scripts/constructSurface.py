import numpy as np
import os

#atype = 'CH1'

if not os.path.isdir("topology"):
        os.mkdir("topology")


layers = 5

atmnm = []
atmnm.append('ASH')
atmnm.append('BSH')
atmnm.append('KSH')
sgm = []
sgm.append(1.000)
sgm.append(1.000)

eps = []
eps.append(1.000)
eps.append(0.800)

mass = 1.000
charge = 0.000

cutoff = 2.5*sgm[0]

xboxn = 20
yboxn = 20

x_patch_level = 1
y_patch_level = 1

offset1 = [3.5*sgm[0], 3.5*sgm[0], 0.5*sgm[0]]
offset2 = [3.5*sgm[0], 3.5*sgm[0], 12.5*sgm[0]]

z_level = sgm[0]/2.0

atoms = []
positions = []

#topfname = f"{base_name}.itp"
#grofname = f"{base_name}.gro"

pos = np.zeros((layers, xboxn, yboxn, 3))

resid = []
for l in range(layers-1):
        resid.append('LJB')
resid.append('LJS')


def genHCPLattice(): #(a, b, c, v1, v2, v3):
        a = 1.000
        b = 1.000
        c = 1.633
        lv = np.array([a,b,c/2])
        v1 = np.array([1, 0.0, 0.0])
        v2 = np.array([1/2.0, np.sqrt(3)/2.0, 0.0])
        v3 = np.array([0.0, 0.0, 1])
        v = []
        v.append(v1*lv)
        v.append(v2*lv)
        v.append(v3*lv)
        v = np.array(v)
        #print(v[:,:])
        init1 = np.array([0.0,0.0,0.0])
        init2 = np.array(v[1,:])
        init = np.array([init1, init2])
        shift1 = np.array([0.0, 0.0, 0.0])
        shift2 = np.array([a/2.0, a/(2*np.sqrt(3)), 0.0])
        ABshift = np.array([shift1, shift2])
        #print(init)
        for l in range(layers):
                for i in range(xboxn):
                        for j in range(yboxn):
                                if(i == 0):
                                        pos[l, 0, j, :] = np.array([init[j%2, 0], init[1, 1]*j, init[j%2, 2]]) + v[2,:]*l
                                else:
                                        pos[l, i, j, :] = pos[l, i-1, j, :] + v[0,:]
        
        for l in range(layers):
                pos[l, :, :, :] = pos[l, :, :, :] + ABshift[l%2, :]


    
def genPatch():
        for l in range(layers):
                for i in range(yboxn):
                        for j in range(xboxn):
                                if(l == layers-1):
                                        if( lattice[j,i] == -1 ):
                                                atomname = atmnm[0]
                                                sigma = sgm[0]
                                        else:
                                                atomname = atmnm[1]
                                                sigma = sgm[1]
                                else:
                                        atomname = atmnm[2]
                                        sigma = sgm[0]
                                atype = 'SH'+atomname[0]
                                ind = l*yboxn*xboxn + i*xboxn + j
                                temp = f'\t{ind+1}\t{atomname}\t{l+1}\t{resid[l]}\t{atype}\t{ind+1}\t{charge:.3f}\t{mass:.3f}\n'
                                atoms.append(temp)
for m in range(81,91):

    atoms = []
    positions = []
    genHCPLattice()
    file_name = f"surface_gen_171.npy"
    lattice = np.load(file_name)
    topfname = f"topology/surface_171.itp"
    grofname = f"topology/surface_171.gro"    
    genPatch()
    #print(atoms)
    for i in range(3):
            pos[:,:,:,i] = pos[:,:,:,i] + offset1[i]
    
    #print(pos)
    
    ubound = np.zeros(3)
    lbound = np.zeros(3)
    
    for i in range(3):
            ubound[i] = np.max(pos[:,:,:,i])
            lbound[i] = np.min(pos[:,:,:,i])
    
    ubound = ubound + offset2
    #print(ubound)
    
    topfile = open(topfname, 'w')
    
    topfile.write("[ moleculetype ]\n")
    topfile.write("; Name nrexcl\n")
    topfile.write(f'LJQ\t{1}\n')
    topfile.write('\n')
    topfile.write('[ atoms ]\n')
    topfile.write(';\tnr\ttype\tresnr\tresid\tatom\tcgnr\tcharge\tmass\n')
    
    for line in range(layers*yboxn*xboxn):
            topfile.write(atoms[line])
    topfile.close()
    
    
    grofile = open(grofname, 'w')
    grofile.write("SAM Surface, t= 0.0\n")
    grofile.write(f'{xboxn*yboxn*layers}\n')
    for l in range(layers):
            for i in range(yboxn):
                    for j in range(xboxn):
                            if(l == layers-1):
                                    if( lattice[j,i] == -1 ):
                                            atomname = atmnm[0]
                                    else:
                                            atomname = atmnm[1]
                            else:
                                    atomname = atmnm[2]
                            atype = 'SH'+atomname[0]
                            ind = l*yboxn*xboxn + i*xboxn + j
                            temp=f'{l+1:5d}{resid[l]:5s}{atype:5s}{ind+1:5d}{pos[l,j,i,0]:8.3f}{pos[l,j,i,1]:8.3f}{pos[l,j,i,2]:8.3f}'
                            grofile.write(temp+'\n')
    grofile.write(f' {ubound[0]:.3f} {ubound[1]:.3f} {ubound[2]:.3f}')
    grofile.close()
    
    surface_dim = np.zeros((3,2))
    box_dim = np.zeros((3,2))
    
    for i in range(3):
            for j in range(2):
                    if(j%2 == 0):
                            surface_dim[i,j] = np.min(pos[-1,:,:,i])
                            box_dim[i, j] = lbound[i]
                    if(j%2 == 1):
                            surface_dim[i,j] = np.max(pos[-1,:,:,i])
                            box_dim[i, j] = ubound[i]
    
    np.save("surface_dim.npy", surface_dim)
    np.save("box_dim.npy", box_dim)
    
    posresfname = "topology/posresSurf.itp"
    
    posresfile = open(posresfname, "w")
    
    posresfile.write("; position restraints for LJ block\n")
    posresfile.write("\n")
    posresfile.write("[ position_restraints ]\n")
    posresfile.write(";  i funct       fcx        fcy        fcz\n")
    
    for i in range(xboxn*yboxn*layers):
            temp=f'{i+1}\t{1}\t{10000}\t{10000}\t{10000}\n'
            posresfile.write(temp)
    
    posresfile.close()
