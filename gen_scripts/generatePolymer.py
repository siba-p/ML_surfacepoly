import numpy as np
import os


if not os.path.isdir("polymer"):
        os.mkdir("polymer")


#atype = 'CH1'
resid = 'KGP'

atnm = []
atnm.append('ACH')
atnm.append('BCH')

chg = []
chg.append(0.000)
chg.append(0.000)

d = np.array([+1,0,0])

mass = 1.000
sigma = 1.000
bondlen = 1.5*sigma
bondconst = 30.0
Nmon = 40

offset = [2*sigma, 2*sigma, 2*sigma]

#atoms = []
#positions = []
for m in range(1,101):
   atoms = []
   positions = []
   fname = f"polymer_{m}.npy"
   topfname = f"polymer/polymer_{m}.itp"
   grofname = f"polymer/polymer_{m}.gro"
   
   surface_dim = np.load("surface_dim.npy")
   box_dim = np.load("box_dim.npy")
   
   lattice = np.load(fname)
   def genSeq(i):
           global lattice
           #'AAAAAAAAAAAAAAAAAAAA'
           if(lattice[i] == -1):
                   return 0
           elif(lattice[i] == 1):
                   return 1
   
   for i in range(Nmon):
           ind = genSeq(i)
           atmnm = atnm[ind]
           charge = chg[ind]
           atype = 'CH'+atmnm[0]
           temp = f'\t{i+1}\t{atmnm}\t{1}\t{resid}\t{atype}\t{i+1}\t{charge:.3f}\t{mass:.3f}\n'
           atoms.append(temp)
           if(i == 0):
                   temp = np.array(offset)
           else:
                   if((i-1)%4 == 0 or i%4 == 0):
                           d[2] = +1
                   else:
                           d[2] = -1
                   if((i+1)%4 == 0 or i%4 == 0):
                           d[1] = -1
                   else:
                           d[1] = +1
   
                   v = 0.333*bondlen*d/np.sqrt(sum(d**2))
   
                   temp = np.array(positions[-1][:]) + v
           positions.append(temp)
   
   topfile = open(topfname, 'w')
   
   topfile.write("[ moleculetype ]\n")
   topfile.write("; Name nrexcl\n")
   topfile.write(f'{resid}\t{1}\n')
   topfile.write('\n')
   topfile.write('[ atoms ]\n')
   topfile.write(';\tnr\ttype\tresnr\tresid\tatom\tcgnr\tcharge\tmass\n')
   
   for i in range(Nmon):
           topfile.write(atoms[i])
   
   
   topfile.write('\n')
   topfile.write('[ bonds ]\n')
   topfile.write(';\tai\taj\tfu\tb\tkb\n')
   
   for j in range(1,Nmon):
           temp = f'\t{j}\t{j+1}\t{7}\t{bondlen:.3f}\t{bondconst:.3f}\n'
           topfile.write(temp)
   
   topfile.close()
   
   grofile = open(grofname, 'w')
   grofile.write("Kremer-Grest Polymer, t= 0.0\n")
   grofile.write(f'{Nmon}\n')
   
   Dmax = np.zeros(3)
   
   positions = np.array(positions)
   
   #print(positions)
   #
   #print(positions[:,0])
   #print(positions[:,1])
   #print(positions[:,2])
   
   COM = np.zeros(3)
   Cdisp = np.zeros(3)
   
   Dmax[0] = np.max(positions[:, 0])
   Dmax[1] = np.max(positions[:, 1])
   Dmax[2] = np.max(positions[:, 2])
   
   boxl = np.max(Dmax) + 2*np.max(offset)
   boxl =f' {boxl:.3f}'
   for i in range(3):
           COM[i] = np.sum(positions[:,i])/Nmon
           Cdisp[i] = COM[i] # -boxl/2.0
           positions[:,i] = positions[:,i] - Cdisp[i]
   
   for i in range(Nmon):
           ind = genSeq(i)
           atmnm = atnm[ind]
           charge = chg[ind]
           atype = 'CH'+atmnm[0]
           temp=f'{1:5d}{resid:5s}{atype:5s}{i+1:5d}{positions[i][0]:8.3f}{positions[i][1]:8.3f}{positions[i][2]:8.3f}'
           grofile.write(temp+'\n')
   #
   #grofile.write(f'{boxl:10.3f} {boxl:10.3f} {boxl:10.3f}')
   grofile.write(f'{boxl} {boxl} {boxl}')
   grofile.close()
