foldername="COLVAR-K500_s1p1"
kappa=500.0
restmin=4.7
restmax=10.4
drest=0.1
ext=_500
MDP="/scratch/siba/ML_go/MDP"
rm -rf ${foldername}
mkdir ${foldername}

#cp s1p1.gro ref.gro

for i in $(seq ${restmin} ${drest} ${restmax})
do

rm -rf $i$ext
mkdir $i$ext

echo -e ${foldername}'/COLVAR_'${i}' \t '${i}' \t '${kappa}' \t 0' >> ${foldername}/metadatafile

cat >${i}${ext}/plumed.dat << EOF
UNITS LENGTH=nm
g1: COM ATOMS=2001-2040
g2: POSITION ATOM=g1


restraint: RESTRAINT ARG=g2.z AT=$i KAPPA=$kappa

PRINT STRIDE=10 ARG=g2.z,restraint.bias FILE=COLVAR_${i}
# the end of plumed input
# ENDMETA
EOF

cd ${i}${ext}

gmx_mpi grompp -f $MDP/eqb.mdp -c ../s1p1.gro -r ../s1p1.gro -o eqb.tpr -p ../topol.top -v
gmx_mpi mdrun -deffnm eqb -plumed plumed.dat -nb cpu -ntomp 4
gmx_mpi grompp -f $MDP/prod.mdp -c eqb.gro -r ../s1p1.gro -o prod.tpr -p ../topol.top -v
gmx_mpi mdrun -deffnm prod -plumed plumed.dat -nb cpu -ntomp 4
cd ..

cp ${i}${ext}/COLVAR_$i ${foldername}/COLVAR_$i
#cp ${i}/prod.gro ref.gro

done
