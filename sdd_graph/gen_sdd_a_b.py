#!/usr/bin/env python

from time import time
import sys
import pickle
import sdd

def neighbors(point,cols):
    """Get the W, S, and SW neighbors of a point"""
    return [point,point-1, point+cols, point+cols-1]

if __name__ == '__main__':
    """Create a structure that represents all paths going from a group of startpoints to a group of endpoints.

    The start point given by the user is the NE point of a group of 4 points
    The end point given by the user is the NE point of a group of 4 points
        The other 3 points are the ones that are one step W, one step S, and two steps SW.

    """


    if len(sys.argv) != 5:
        print "usage: %s [GRID-M] [GRID-N] [STARTPOINT] [ENDPOINT]" % sys.argv[0]
        exit(1)
    dim = (int(sys.argv[1]),int(sys.argv[2]))
    rows = dim[0]
    cols = dim[1]
    dimension = (dim[0]-1,dim[1]-1)
    startpoint = int(sys.argv[3])
    endpoint = int(sys.argv[4])

    starts = neighbors(startpoint, cols)
    ends = neighbors(endpoint, cols)

    from graphillion import GraphSet
    import graphillion.tutorial as tl
    universe = tl.grid(*dimension)
    GraphSet.set_universe(universe)

    paths = GraphSet()

    for start in starts:
        for end in ends:
            paths = GraphSet.union(paths,GraphSet.paths(start,end))

    for i in range(4):
        tl.draw(paths.choice())
