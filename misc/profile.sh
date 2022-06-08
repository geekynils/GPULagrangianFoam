#!/bin/bash

source $HOME/OpenFOAM/OpenFOAM-1.6-ext/etc/bashrc
cd $FOAM_RUN/GPULagrangianFoam/testCases/torus
calcLambdas -data testData
