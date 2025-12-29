import numpy as np

from constructSurface import sgm

box_dim = np.load("box_dim.npy")
surface_dim = np.load("surface_dim.npy")

polyCOMpos = [(surface_dim[0,0]+surface_dim[0,1])/2.0,(surface_dim[1,0]+surface_dim[1,1])/2.0,surface_dim[2,1]+ 1.5*sgm[0]]

def readGroLine(line):
        currInd = 0
        rid = int(line[currInd:currInd+5])
        currInd = currInd+5
        rname = line[currInd:currInd+5]
        currInd = currInd+5
        aname = line[currInd:currInd+5]
        currInd = currInd+5
        aid = int(line[currInd:currInd+5])
        currInd = currInd+5
        x = float(line[currInd:currInd+8])
        currInd = currInd+8
        y = float(line[currInd:currInd+8])
        currInd = currInd+8
        z = float(line[currInd:currInd+8])
        currInd = currInd+8

        return rid, rname, aname, aid, x, y, z

def readGroFile(fname):

        data = open(fname, 'r')

        data.readline()
        natoms = int(data.readline())


        resid = []
        resname = []
        atomid = []
        atomname = []
        pos = []
        

        for atoms in range(natoms):
                t1, t2, t3, t4, t5, t6, t7 = readGroLine(data.readline())
                resid.append(t1)
                resname.append(t2)
                atomname.append(t3)
                atomid.append(t4)
                pos.append([t5, t6, t7])

        data.close()

        resid = np.array(resid)
        resname = np.array(resname)
        atomid = np.array(atomid)
        atomname = np.array(atomname)
        pos = np.array(pos)

        #return resid, resname, atomname, atomid, pos
        return [resid, resname, atomname, atomid, pos]
for o in range(81,91):
   for m in range(1,91):
    
      Adata = readGroFile(f"topology/surface_191.gro")
      Bdata = readGroFile(f"polymer/polymer_90.gro")
      
      Alen = len(Adata[0])
      Blen = len(Bdata[0])
      
      
      Fresid = []
      Fresname = []
      Fatomid = []
      Fatomname = []
      Fpos = []
      
      for l in range(Alen+Blen):
              if(l < Alen):
                      Fresid.append(Adata[0][l])
                      Fresname.append(Adata[1][l])
                      Fatomname.append(Adata[2][l])
                      Fatomid.append(Adata[3][l])
                      Fpos.append(Adata[4][l])
              else:
                      Fresid.append(Bdata[0][l-Alen]+Adata[0][-1])
                      Fresname.append(Bdata[1][l-Alen])
                      Fatomname.append(Bdata[2][l-Alen])
                      Fatomid.append(Bdata[3][l-Alen]+Adata[3][-1])
                      temp = (np.array(Bdata[4][l-Alen]) + np.array(polyCOMpos)).tolist()
                      Fpos.append(temp)
      
      #writing to new gro file
      boxfname = f"topology/s191p91.gro"
      
      boxfile = open(boxfname, 'w')
      boxfile.write("Simulation Box, t= 0.0\n")
      boxfile.write(f'{Alen+Blen}\n')
      box_z = 25 
      for l in range(Alen+Blen):
              temp=f'{Fresid[l]:5d}{Fresname[l]:5s}{Fatomname[l]:5s}{Fatomid[l]:5d}{Fpos[l][0]:8.3f}{Fpos[l][1]:8.3f}{Fpos[l][2]:8.3f}'
              boxfile.write(temp+'\n')
      
      boxfile.write(' {:.3f} {:.3f} {:.3f}'.format(box_dim[0,1],box_dim[1,1],box_z)) 
     # boxfile.write(f' {box_dim[0,1]:.3f} {box_dim[1,1]:.3f} {box_dim[2,1]:.3f}')
      
      boxfile.close()
      
#      restraintfname = "restraint_2.gro"
      
#      restraintfile = open(restraintfname, 'w')
#      restraintfile.write("Restraint File for Box\n")
#      restraintfile.write(f'{Alen+Blen}\n')
      
#      for l in range(Alen+Blen):
#              if(l < Alen):
#                      temp=f'{Fresid[l]:5d}{Fresname[l]:5s}{Fatomname[l]:5s}{Fatomid[l]:5d}{Fpos[l][0]:8.3f}{Fpos[l][1]:8.3f}{Fpos[l][2]:8.3f}'
#                      restraintfile.write(temp+'\n')
#              else:
#                      temp=f'{Fresid[l]:5d}{Fresname[l]:5s}{Fatomname[l]:5s}{Fatomid[l]:5d}{polyCOMpos[0]:8.3f}{polyCOMpos[1]:8.3f}{polyCOMpos[2]:8.3f}'
#                      restraintfile.write(temp+'\n')
      
      
#      restraintfile.write(f' {box_dim[0,1]:.3f} {box_dim[1,1]:.3f} {box_dim[2,1]:.3f}')
      
#      restraintfile.close()
      
      
#      dim = []
#      dim.append('x')
#      dim.append('y')
#      dim.append('z')
      
#      r_constraint = np.zeros(3)
      
#      for i in range(3):
      
#              if(i == 2):
#                      r_constraint[i] = np.abs(surface_dim[i,1]-polyCOMpos[i])+sgm[0]
#              else:
#                      r_constraint[i] = np.abs(surface_dim[i,1] - surface_dim[i,0])/2.0
      
#              restfile = open(f'topology/restraint{dim[i]}.itp', "w")
      
#              restfile.write(f';Position restraint for Kremer-Grest polymer perp to {dim[0]}\n')
#              restfile.write("[ position_restraints ]\n")
#              restfile.write("\n")
#              restfile.write(";\ti\tfunct\tg\tr(nm)\tk\n")
#              restfile.write('\n')
#              for j in range(Blen):
#                      restfile.write(f'\t{j+1}\t{2}\t{3+i}\t{r_constraint[i]:.3f}\t{50000}\n')
#              restfile.close()
