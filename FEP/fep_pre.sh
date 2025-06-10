#!/bin/bash 

fep_file_generate_deLP(){
    state_A=$1
    name_A=`basename ${state_A} .tgz`
    state_B=$2
    name_B=`basename ${state_B} .tgz`
    mkdir temp
    tar -zxvf ${state_A} -C temp
    cp ./temp/charmm-gui*/ligandrm.pdb ${name_A}.pdb
    sed -i '/LP/d' ${name_A}.pdb
    cp ./temp/charmm-gui*/lig/lig.prm ${name_A}.prm
    cp ./temp/charmm-gui*/lig/lig.rtf ${name_A}.rtf
    sed -i '/LP/d' ${name_A}.rtf
    rm -r temp
    mkdir temp
    tar -zxvf ${state_B} -C temp
    cp ./temp/charmm-gui*/ligandrm.pdb ${name_B}.pdb
    sed -i '/LP/d' ${name_B}.pdb
    cp ./temp/charmm-gui*/lig/lig.prm ${name_B}.prm
    cp ./temp/charmm-gui*/lig/lig.rtf ${name_B}.rtf
    sed -i '/LP/d' ${name_B}.rtf
    rm -r temp ${state_A} ${state_B}
}

input=$*
fep_file_generate_deLP ${input}
