#!/bin/bash

starts=(18 15 90 62 32 17 77 38 39 23 47)
ends=(73 88 22 6 89 93 19 75 81 86 71)

for ((run=0;run<11;run++))
do
	python buildGraphs.py 10 10 ${starts[$run]} ${ends[$run]}
done
