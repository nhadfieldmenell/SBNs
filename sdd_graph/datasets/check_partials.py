#!/usr/bin/python

import sys

rows = sys.argv[1]
cols = sys.argv[2]
partials = open("partials_%d_%d.txt" % (rows,cols),'r')
lines = partials.readlines()
for i in range(len(lines)):
    spots = map(int,lines[i][:-1].split(','))
    for j in range(len(spots)):
        if spots[j] == 0:
            print "0: %d" % (j+1)
        if spots[j] == 1:
            print "1: %d" % (j+1)

