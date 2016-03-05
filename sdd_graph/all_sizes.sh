#!/bin/bash

for i in 2 3 4 5 6 7 8 9 10 11 12
do 
    echo $(((i*i)/2))
    python test_grids_minato.py $i $i $(((i*i)/2))
    python minato2sdd.py $i $i
    python print_sdd.py $i $i
done
