#!/usr/bin/env python

from time import time
import sys
import pickle

#import sdd

def draw_grid(model,dimension):
    for i in xrange(dimension):
        for j in xrange(dimension):
            sys.stdout.write('.')
            if j < dimension-1:
                index = i*(dimension-1) + j + 1
                sys.stdout.write('-' if model[index] else ' ')
        sys.stdout.write('\n')
        if i < dimension-1:
            for j in xrange(dimension):
                index = dimension*(dimension-1) + j*(dimension-1) + i + 1
                sys.stdout.write('|' if model[index] else ' ')
                sys.stdout.write(' ')
        sys.stdout.write('\n')

def print_grids(alpha,dimension,manager):
    #import pdb; pdb.set_trace()
    from inf import models
    #var_count = 2*dimension*(dimension-1)
    #var_count = 2*dimension*(dimension-1) + dimension*dimension
    print "COUNT:", sdd.sdd_model_count(alpha,manager)
    for model in models.models(alpha,sdd.sdd_manager_vtree(manager)):
        print models.str_model(model,var_count=var_count)
        draw_grid(model,dimension)

def save_grid_graph(filename,graph):
    with open(filename,'wb') as output:
        pickle.dump(graph,output)

########################################
# MAIN
########################################

# 2,14,322,23858,5735478,4468252414 ???
# 2,12,184,8512,
# 2
# 12
# 184
# 8512
# 1262816
# 575780564
# 789360053252
# 3266598486981642
# 41044208702632496804
# 1568758030464750013214100
# 182413291514248049241470885236 
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "usage: %s [GRID-M] [GRID-N]" % sys.argv[0]
        exit(1)
    dim = (int(sys.argv[1]),int(sys.argv[2]))
    dimension = (dim[0]-1,dim[1]-1)
    #dimension = (1,1)

    from graphillion import GraphSet
    import graphillion.tutorial as tl
    universe = tl.grid(*dimension)
    GraphSet.set_universe(universe)

    start,goal = 1,(dimension[0]+1)*(dimension[1]+1)
    #create an empty GraphSet
    paths = GraphSet()
    midpoint = 3
    #paths = GraphSet.paths(start, goal)
    for i in range(start,goal):
        for j in range(i+1,goal+1):
            if i != midpoint and j != midpoint:
                paths = GraphSet.union(paths,GraphSet.paths(i,j))

    pathsThruMidpoint = paths.including(midpoint)
    print pathsThruMidpoint.len()
    #tl.draw(pathsThruMidpoint.choice())
    print GraphSet
    print paths.len()

    #dim = (dimension[0]+1,dimension[1]+1)
    #""" AC: SAVE ZDD TO FILE
    f = open("asdf-%d-%d.zdd" % dim,"w")
    pathsThruMidpoint.dump(f)
    f.close()
    #"""

    """ AC: CREATE GRAPH?
    nodes = [None] + [ (x,y) for x in xrange(dim[0]) for y in xrange(dim[1]) ]
    from collections import defaultdict
    graph = defaultdict(list)
    for index,edge in enumerate(paths.universe()):
        x,y = edge
        x,y = nodes[x],nodes[y]
        graph[x].append( (index+1,y) )
        graph[y].append( (index+1,x) )

    #graph_filename = "asdf-%d-%d.graph.pickle" % dim
    #save_grid_graph(graph_filename,graph)
    """


    #sdd_filename = "output/paths/paths-%d.sdd" % dimension
    #sdd_vtree_filename = "output/paths/paths-%d.vtree" % dimension
    #graph_filename = "output/paths/paths-%d.graph.pickle" % dimension
    #sdd.sdd_save(sdd_filename,alpha)
    #sdd.sdd_vtree_save(sdd_vtree_filename,sdd.sdd_manager_vtree(manager))

    #graph = _node_to_edge_map(dimension)
    #save_grid_graph(graph_filename,graph)

