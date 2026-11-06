#!/bin/bash
##This script will generate metadtafile in all directories
restmin=4.7
restmax=10.4
drest=0.1
file="COLVAR-K500_"
kappa=500
ext="_500"
total=92
#i=11
spin="/-\|"
for i in $(seq 21  30); do 
    if [ ! -d "surface${i}" ]; then
        echo "Directory surface${i} not found."
        exit 1
    fi
    cd surface${i}
    
    for j in $(seq 0 100); do
        if [ ! -d "s${i}p${j}" ]; then
#            echo "Directory s${i}p${j} not found" >> error.dat
            continue
        fi
        
        printf "\rprogress: %s%d/%d" "${spin:j%4:1}" "$j" "$total"
        cd s${i}p${j} || { echo "Failed to enter directory s${i}p${j}"; continue; }
       
        foldername="${file}s${i}p${j}"
#        mkdir -p "$foldername"
#        rm -rf "$foldername"
        k=$restmin
        while (( $(echo "$k <= $restmax" | bc -l) )); do
           # printf "%s\t%.1f\t%d\t%d\n" "$(pwd)/${foldername}/COLVAR_${k}" "${k}" "${kappa}" 0 >> "${foldername}/metadata"
            source_file="${k}${ext}/COLVAR_${k}"
	    source_file2="${k}${ext}/prod.gro"
            if [ ! -e "$source_file2" ]; then
#                     
#	        cp "$source_file" "${foldername}/COLVAR_${k}"
#            else
                echo "Warning: File $source_file2 not found in s${i}p${j}/${k}${ext}" 
#                cd ${k}${ext}
#		gmx_mpi mdrun -deffnm prod -plumed plumed.dat -nb cpu -ntomp 4 -v
#		cd ..
            fi
#    
            k=$(echo "$k + $drest" | bc)
        done
    
        cd ..
    done
    echo -e "completed!"
    cd ..
done
