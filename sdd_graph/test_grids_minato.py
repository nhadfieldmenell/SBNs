#!/usr/bin/env python

from time import time
import sys
import pickle
import sdd

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
    var_count = 2*dimension*(dimension-1)
    #var_count = 2*dimension*(dimension-1) + dimension*dimension
    #var_count = 2*dimension[0]*dimension[1] - dimension[0] - dimension[1]
    
    print "COUNT:", sdd.sdd_model_count(alpha,manager)
    for model in models.models(alpha,sdd.sdd_manager_vtree(manager)):
        print models.str_model(model,var_count=var_count)
        draw_grid(model,dimension)

def save_grid_graph(filename,graph):
    with open(filename,'wb') as output:
        pickle.dump(graph,output)

def print_edge_numbering(universe,rows,cols):
    tuple2edge = {}
    for i in range(len(universe)):
        tuple2edge[universe[i]] = i+1

    for row in range(rows):
        for col in range(cols-1):
            node = row*cols+col+1
            next_node = node+1
            edge = (node,next_node)
            edge_num = tuple2edge[edge]
            sys.stdout.write("O ")
            if edge_num < 100:
                sys.stdout.write(" ")
            sys.stdout.write("%d  " % edge_num)
            if edge_num < 10:
                sys.stdout.write(" ")
        sys.stdout.write("O\n\n")
        if row == rows-1:
            break
        for col in range(cols):
            node = row*cols+col+1
            next_node = node+cols
            edge = (node,next_node)
            edge_num = tuple2edge[edge]
            sys.stdout.write("%d    " % edge_num)
            if edge_num < 100:
                sys.stdout.write(" ")
                if edge_num < 10:
                    sys.stdout.write(" ")
        sys.stdout.write("\n\n")


def create_edge_to_index(paths,dim):
    tuple2edge = {}
    universe = paths.universe()
    for i in range(len(universe)):
        tuple2edge[universe[i]] = i
    edge_filename = "graphs/edge-nums-%d-%d.pickle" % dim
    with open(edge_filename,'wb') as output:
        pickle.dump(tuple2edge,output)


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
    if len(sys.argv) != 4:
        print "usage: %s [GRID-M] [GRID-N] [MIDPOINT]" % sys.argv[0]
        exit(1)
    dim = (int(sys.argv[1]),int(sys.argv[2]))
    dimension = (dim[0]-1,dim[1]-1)
    midpoint = int(sys.argv[3])
    #dimension = (1,1)

    from graphillion import GraphSet
    import graphillion.tutorial as tl
    universe = tl.grid(*dimension)
    GraphSet.set_universe(universe)

    start,goal = 1,(dimension[0]+1)*(dimension[1]+1)
    #create an empty GraphSet
    paths = GraphSet()
    paths_no_mp = GraphSet()
    for i in range(start,goal):
        print i
        for j in range(i+1,goal+1):
            #paths = GraphSet.union(paths,GraphSet.paths(i,j))
            """Exclude midpoint"""
            if i != midpoint and j != midpoint:
                paths_no_mp = GraphSet.union(paths_no_mp,GraphSet.paths(i,j))
            paths = GraphSet.union(paths,GraphSet.paths(i,j))

    pathsThruMidpoint = paths.including(midpoint)
    pathsNoMidpoint = paths_no_mp.including(midpoint)

    #tl.draw(pathsThruMidpoint.choice())
    print "number of paths through midpoint: " + str(pathsThruMidpoint.len())
    print "number of paths without stopping at midpoint: " + str(pathsNoMidpoint.len())
    print_edge_numbering(paths.universe(),dim[0],dim[1])
    #print paths.universe()
    #for p in pathsThruMidpoint:
    #    print p

    #dim = (dimension[0]+1,dimension[1]+1)
    #""" AC: SAVE ZDD TO FILE
    f = open("graphs/asdf-%d-%d.zdd" % dim,"w")
    pathsThruMidpoint.dump(f)
    f.close()
    #"""
    #""" AC: SAVE ZDD TO FILE
    f = open("graphs/asdf-no-mp-%d-%d.zdd" % dim,"w")
    pathsNoMidpoint.dump(f)
    f.close()
    #"""

    """ AC: SAVE GRAPH """
    nodes = [None] + [ (x,y) for x in xrange(dim[0]) for y in xrange(dim[1]) ]
    from collections import defaultdict
    graph = defaultdict(list)
    for index,edge in enumerate(pathsThruMidpoint.universe()):
        x,y = edge
        x,y = nodes[x],nodes[y]
        graph[x].append( (index+1,y) )
        graph[y].append( (index+1,x) )
    graph_filename = "graphs/asdf-%d-%d.graph.pickle" % dim

    nodesNoMP = [None] + [ (x,y) for x in xrange(dim[0]) for y in xrange(dim[1]) ]
    graphNoMP = defaultdict(list)
    for index,edge in enumerate(pathsNoMidpoint.universe()):
        x,y = edge
        x,y = nodesNoMP[x],nodesNoMP[y]
        graphNoMP[x].append( (index+1,y) )
        graphNoMP[y].append( (index+1,x) )
    graphNoMP_filename = "graphs/asdf-no-mp-%d-%d.graph.pickle" % dim

    # save to file
    import pickle
    with open(graph_filename,'wb') as output:
        pickle.dump(graph,output)

    with open(graphNoMP_filename,'wb') as output:
        pickle.dump(graphNoMP,output)

    tuple2edge = {}
    universe = paths.universe()
    for i in range(len(universe)):
        tuple2edge[universe[i]] = i
    edge_filename = "graphs/edge-nums-%d-%d.pickle" % dim
    with open(edge_filename,'wb') as output:
        pickle.dump(tuple2edge,output)

    #sdd_filename = "output/paths/paths-%d.sdd" % dimension
    #sdd_vtree_filename = "output/paths/paths-%d.vtree" % dimension
    #graph_filename = "output/paths/paths-%d.graph.pickle" % dimension
    #sdd.sdd_save(sdd_filename,alpha)
    #sdd.sdd_vtree_save(sdd_vtree_filename,sdd.sdd_manager_vtree(manager))

    #graph = _node_to_edge_map(dimension)
    #save_grid_graph(graph_filename,graph)

