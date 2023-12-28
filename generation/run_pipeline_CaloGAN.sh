#!/bin/bash                                                                     

jobnum=$1
CALOGANDIR=/het/p4/ian/miniconda3/envs/py2env/CaloGAN/generation
RETURNDIR=/het/p4/ian/miniconda3/envs/py2env/CaloGAN/generation
cd $CALOGANDIR
#source cfg/lxplus-env.sh                                                       
#make                                                                           
mkdir ${jobnum}
cd ${jobnum}
generate -m ../cfg/run2.mac
mv plz_work_kthxbai_t0.root ${RETURNDIR}/${jobnum}_t0.root
#mv plz_work_kthxbai_t1.root ${RETURNDIR}/${jobnum}_t1.root                     
cd ../
rm -r ${jobnum}
