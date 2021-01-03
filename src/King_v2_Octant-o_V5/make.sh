#!/bin/bash
if [ $# -eq 2 ]
    then
        if [ "$1" == "mkl" ]
            then
                echo "Use MKL to compile."
                export MKL_FLAG="ON"
        elif [ "$1" == "lapack" ]
            then
                echo "Use LAPACK to compile."
                export MKL_FLAG="OFF"
        else
            echo "Must select mkl/lapack!"
            exit 1
        fi
        if [ "$2" == "oct" ]
            then
                echo "Turn on OCTANT."
                export OCTANT_FLAG="ON"
        elif [ "$2" == "nooct" ]
            then
                echo "Turn off OCTANT."
                export OCTANT_FLAG="OFF"
        else
            echo "Must select oct/nooct!"
            exit 1
        fi
        make -j8
elif [ "$1" == "clean" ] 
    then
        make clean
else
    echo -e "Usage:\n\tsh make.sh mkl/lapack oct/nooct to use mkl/lapack to compile, with OCTANT_DECOMPOSE on/off.\n\tsh make.sh clean for removing all .o and executation files."
fi      
unset MKL_FLAG
unset OCTANT_FLAG
