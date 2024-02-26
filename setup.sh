#!/bin/bash

conda create -n graphmapping2 python=3.10 ipykernel numpy networkx pandas pyscipopt matplotlib scipy pulp pyyaml yaml pytorch::pytorch gymnasium -c conda-forge -y

mkdir data
mkdir data/problems
mkdir data/solutions
mkdir data/results
mkdir data/logs
mkdir data/temp