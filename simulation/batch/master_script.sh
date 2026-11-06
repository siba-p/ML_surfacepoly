#!/bin/bash


DIR=`pwd`

MDP="$DIR/MDP"
CODE="$DIR/build_code"

fracA=0.1
fracB=1.0 

cd parent_dir
for i in $(seq $fracA $fracA $fracB);do                         ## starting of first do loop
        index=$(echo "$i * 10" | bc | awk '{printf "%d", $0}')
	if [ -d $i ];then
		echo "directory found named as $i"
		rm -r $i
	fi	
                echo "creating directory"
		mkdir $i
	
        cd $i
	for j in $(seq $(((index - 1) * 10 + 1)) $((index * 10)));do ## 2nd do loop
		if [ -d surface${j} ];then
                    echo "directory found named as surface${j}"
                    rm -r surface${j}
		fi    
                    echo "creating directory"
 		    mkdir surface${j}
                
    		
		cd surface${j}
		cp $DIR/topology/restraint* $DIR/topology/posresSurf.itp ./
		cp $CODE/topology/${i}/surface/surface_${j}.itp ./
		echo "entering surface${j} within $i"
		for k in $(seq 1 90);do                             ## 3rd do loop
			mkdir s${j}p${k}
                       for m in $(seq 1 10);do
                        cp $CODE/surface_${m}.npy s${j}p${k}/
                        cp -r $DIR/topology/wham s${j}p${k}/
		        cp $DIR/pmf.sh s${j}p${k}/
		 
		        cp $CODE/topology/${i}/surface${j}/s${j}p${k}.gro s${j}p${k}
		        cp $CODE/polymer/polymer_${k}.itp s${j}p${k}
		        cp $DIR/topology/topol.top s${j}p${k} 	
                        # done
	                cp $CODE/topology/surface_${j}.itp ./
	                cd s${j}p${k}
			sed -i "s/COLVAR-K500/COLVAR-K500_s${j}p${k}/g" pmf.sh
		        cp $DIR/topology/restraint* $DIR/topology/posresSurf.itp ./ 
		        sed -i "s/XXX/s${j}p${k}/g" pmf.sh
                        sed -i "s/polymer.itp/polymer_${k}.itp/g" topol.top
                        sed -i "s/surface.itp/surface_${j}.itp/g" topol.top
	               cd ..	       
                done
                cd ..
        done	
        cd ..
done
cd ..

