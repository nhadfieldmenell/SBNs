#!/usr/bin/env python

from time import time
import sys
import pickle
import sdd
import test_grids_minato as minato

def all_paths(dimension,dim):
    from graphillion import GraphSet
    import graphillion.tutorial as tl
    start,goal = 1,(dimension[0]+1)*(dimension[1]+1)
    
    universe = tl.grid(*dimension)
    GraphSet.set_universe(universe)
    paths = GraphSet()
    for i in range(start,goal):
        for j in range(i+1,goal+1):
            paths = GraphSet.union(paths,GraphSet.paths(i,j))

    f = open("graphs/general_ends-%d-%d.zdd" % (dim[0],dim[1]),"w")
    paths.dump(f)
    f.close()

    nodes = [None] + [ (x,y) for x in xrange(dim[0]) for y in xrange(dim[1]) ]
    from collections import defaultdict
    graph = defaultdict(list)
    for index,edge in enumerate(paths.universe()):
        x,y = edge
        x,y = nodes[x],nodes[y]
        graph[x].append( (index+1,y) )
        graph[y].append( (index+1,x) )
    graph_filename = "graphs/general_ends-%d-%d.graph.pickle" % (dim[0],dim[1])

    with open(graph_filename,'wb') as output:
        pickle.dump(graph,output)

def main():
    """Create a structure that represents all paths going from a group of startpoints to a group of endpoints.

    The start point given by the user is the NE point of a group of 4 points
    The end point given by the user is the NE point of a group of 4 points
        The other 3 points are the ones that are one step W, one step S, and two steps SW.

    """


    if len(sys.argv) != 3:
        print "usage: %s [GRID-M] [GRID-N]" % sys.argv[0]
        exit(1)
    dim = (int(sys.argv[1]),int(sys.argv[2]))
    rows = dim[0]
    cols = dim[1]
    dimension = (dim[0]-1,dim[1]-1)

    from graphillion import GraphSet
    import graphillion.tutorial as tl

    all_paths(dimension,dim)

if __name__ == '__main__':
    main()
