#!/usr/bin/env python

from time import time
import sys
import pickle
import sdd
import test_grids_minato as minato

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

    
    """ AC: SAVE ZDD TO FILE """
    f = open("graphs/start_end-%d-%d-%d-%d.zdd" % (dim[0],dim[1],startpoint,endpoint),"w")
    pathsThruMidpoint.dump(f)
    f.close()

    
    """ AC: SAVE GRAPH """
    nodes = [None] + [ (x,y) for x in xrange(dim[0]) for y in xrange(dim[1]) ]
    from collections import defaultdict
    graph = defaultdict(list)
    for index,edge in enumerate(paths.universe()):
        x,y = edge
        x,y = nodes[x],nodes[y]
        graph[x].append( (index+1,y) )
        graph[y].append( (index+1,x) )
    graph_filename = "graphs/start_end-%d-%d-%d-%d.graph.pickle" % (dim[0],dim[1],startpoint,endpoint)

    # save to file
    import pickle
    with open(graph_filename,'wb') as output:
        pickle.dump(graph,output)
