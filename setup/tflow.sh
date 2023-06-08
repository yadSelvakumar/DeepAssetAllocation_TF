#!/bin/bash

cd $(dirname $BASH_SOURCE)

input() {
    echo '----------------------------------------------'
    echo "$1"
    while [ "$ans" != 'y' ] && [ "$ans" != 'n' ] ; do
        read -p 'y/n: ' ans
    done
    [ "$ans" == 'y' ] && make $2
    ans=0
}

input 'Update Ubuntu?' update

input 'Download conda?' condadownload

source ~/.bashrc && conda -V

input 'Create conda env (tf)?' condaenv

input 'Install conda-forge?' condaforge

input 'Set conda cuda path?' condapath

input 'Pip upgrade and install tensorflow?' tensorflow

input 'Test CPU?' tfcpu

input 'Test GPU?' tfgpu

input 'Install project pip requeriments?' pipinstall

