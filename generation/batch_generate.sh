#!/bin/csh
#$ -j y

#echo "Assuming working directory: $PBS_O_WORKDIR"
#cd "$PBS_O_WORKDIR"
#echo $(pwd)

source cfg/pdsf-env.sh



#GAN_FNAME="generated_file_$SGE_TASK_ID.root"

setenv GAN_FNAME "generated_file_$SGE_TASK_ID.root" 
generate -m cfg/run2.mac
