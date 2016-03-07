#!/usr/bin/python
import sys
from collections import defaultdict
import numpy as np
import decodeGps as dg
import test_graph as tg

class Graph(object):
    """Object that holds edge/node information for a gps graph.

    The graph is partitioned into "squares" according to max/min lat/lon and dimensions.

    Attributes:
        min_lat: the minimum latitude for the graph
        max_lat: the maximum latitude for the graph
        min_lon: the minimum longitude for the graph
        max_lon: the maximum longitude for the graph
        rows: the number of rows for the graph
        cols: the number of columns for the graph
    """

    def __init__(self,fn,min_lat,max_lat,min_lon,max_lon,rows,cols):
        self.lines = fn.readlines()
        self.gps_length = len(self.lines)
        self.trip_id2line_num = {}

        self.rows = rows
        self.cols = cols
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon

        self.lat_step = float((max_lat-min_lat)/rows)
        self.lon_step = float((max_lon-min_lon)/cols)

        self.diags = self.create_diags()
        self.num_edges = self.diags[self.rows+self.cols-2]
        self.node2visited = defaultdict(int) 
        self.node2trip_ids = defaultdict(list)
        self.best_node = 1
        self.best_node_score = 0
        #print self.num_edges

    def node_visit(self,trip_id,coords):
        """Set nodes_visited and node2visited to reflect that the node coords is visited by trip number trip_id.

        Attributes:
            trip_id: integer id for the trip.
            coords: (row,column) that is visited
        """
        if trip_id not in self.trip_id2line_num:
            node_num = self.coords_to_node(coords[0],coords[1])
            self.node2visited[node_num] += 1
            if self.node2visited[node_num] > self.best_node_score:
                self.best_node_score = self.node2visited[node_num]
                self.best_node = node_num
            self.node2trip_ids[node_num].append(trip_id)


    
    def edge_num(self,row1,col1,row2,col2):
        """Determines the edge number between two points.

        The method determines which point comes first.

        This representation
            X 0 X 2 X
            1   3   6
            X 4 X 7 X
            5   8  10
            X 9 X 11X

        Sdd representation
            X 1 X 3 X
            2   4   7
            X 5 X 8 X
            6   9  11
            X 10X 12X

        Attributes:
            row1: row number for first point.
            col1: column number for first point.
            row2: row number for second point.
            col2: column number for second point.

        Returns:
            An integer corresponding to the edge number between two points.
            -1 if the points are not valid neighbors.
        """

        row = row1
        col = col1
        row_n = row2
        col_n = col2
        
        if row2 < row1 or col2 < col1:
            row = row2
            col = col2
            row_n = row1
            col_n = col1
        
        if not (row == row_n and col == col_n - 1 or row == row_n-1 and col == col_n):
            return - 1
        
        edge_number = 0

        if row + col < self.cols - 1:
            if col_n == col + 1: 
                #print "(%d,%d) (%d,%d)" % (row, col, row, col + 1)
                edge_number = self.diags[row + col] + 2 * row
                #edges[edge_number] = 1
            elif row_n == row + 1:
                #print "(%d,%d) (%d,%d)" % (row, col, row + 1, col)
                edge_number = self.diags[row + col] + 1 + 2 * row
                #edges[edge_number] = 1
        else:
            col_dist = self.cols - col - 1
            if col_n == col + 1: 
                #print "(%d,%d) (%d,%d)" % (row, col, row, col + 1)
                edge_number = self.diags[row + col] + 2 * col_dist  - 1
                #edges[edge_number] = 1
            elif row_n == row + 1:
                #print "(%d,%d) (%d,%d)" % (row, col, row + 1, col)
                edge_number = self.diags[row + col] + 2 * col_dist
                #edges[edge_number] = 1

        return edge_number


    def create_diags(self):
        """Creates a diagonals array for the graph

        Diagonals are used in the ordering of edges for our graph representation.
        For a point (i,j), there are diag_full[i+j] total edges that appeared in previous diagonals.

        Returns:
            An array that counts the number of edges occurring before a specified diagonal.
        """

        num_diags = self.rows + self.cols - 2
        diag_counts = [0 for i in range(num_diags)]
        for diag_index in range(num_diags):
            first = (0,0)
            second = (0,0)
            if diag_index < self.rows - 1:
                first = (diag_index+1,0)
            elif diag_index == self.rows - 1:
                first = (diag_index,0)
            else:
                first = (self.rows-1,diag_index-self.rows+1)
            if diag_index < self.cols - 1:
                second = (0,diag_index+1)
            elif diag_index == self.cols - 1:
                second = (0,diag_index)
            else:
                second = (diag_index-self.cols+1,self.cols-1)
            #print str(first) + " " + str(second)
            diag_counts[diag_index] = dist_points(first,second) 
       
        """holds the sum of edges in diagonals previous to a given edge"""
        diag_full = [0 for i in range(num_diags + 1)]
        for i in range(1,num_diags+1):
            diag_full[i] = diag_full[i-1] + diag_counts[i-1]

        print diag_counts
        print diag_full
        return diag_full


    def node_to_coords(self,node_num):
        """determines (row,col) for a given node

        node ordering:
        1 2 3
        4 5 6
        7 8 9

        Attributes:
            node_num: node index according to the numbering shown above

        Returns:
            (row,col): corresponding coordinates.
        """
        row = (node_num - 1) / self.cols
        col = (node_num - 1) % self.cols
        return (row,col)

    def coords_to_node(self,row,col):
        """determines the node index of a coordinate in the path graph

        1 2 3
        4 5 6
        7 8 9

        Attributes:
            row: row number (0 indexed)
            col: col number (0 indexed)

        Returns:
            node number: (1 indexed)
        """
        return row*self.cols + col + 1

    def gps_to_coords(self,lat,lon):
        """Determines the coodinates on the graph corresponding to a given gps point.

        Attributes:
            lat: the latitude of the point
            lon: the longitude of the point

        Returns:
            A pair corresponding to the (row,column) in the graph that holds that gps point
            (-1,-1) if the coordinates are out of scope
        """

        if (lat < self.min_lat or lat > self.max_lat or lon < self.min_lon or lon > self.max_lon):
            return (-1,-1)

        lat_spot = int((self.max_lat-lat)/self.lat_step)
        lon_spot = int((lon-self.min_lon)/self.lon_step)
        return (lat_spot,lon_spot)

def dist_points(x,y):
    """Finds the shortest distance between two points in a grid graph.
        
    Args:
        x: a tuple (i,j)
        y: a tuple (i,j)

    Returns:
        An integer which is the shortest path distance between the points.
    """

    return abs(x[0]-y[0]) + abs(x[1]-y[1])

class Path(object):
    """A path in the graph g. 

    Holds a grid representative of the graph, traversed nodes are True, others are False.
    Creates an edge mapping of all active edges in the graph.
    Any gps coordinates that are outside of the graph are not included.

    Attributes:
        trip_id: int valued trip id for the path in the gps document.
        graph: the graph object over which we are creating the path.
        fn: file descriptor for the original csv gps file.
        line_num: the initial line of the trip (default to -1 if first line unknown).
        lines: the lines of the gps file held in an array.
        gps_length: the number of lines from the gps file.
        path: a 2-d numpy array representing the path on the graph.
            matrix positions are 1 if the path traverses that node, 0 otherwise
        edges: an array that holds a 1 if the edge corresponding to the index 
            is used, 0 otherwise.
    """
    #@profile
    def __init__(self,trip_id,graph,fn,line_num=-1):
        self.graph = graph
        self.trip_id = trip_id
        self.line_num = line_num
        self.bad_graph = False
        self.next_line = 0
        self.line_num = self.find_path()
        self.path,self.edges,self.good = self.create_path()
        #print self.trip_id
        #print self.path
        #self.edges_alt = self.path_to_edges()
        """this changes the edge list to be 1-indexed for draw_grids"""
        self.draw_edges = self.edges[:]
        self.draw_edges.insert(0,0)
        #self.draw_edges_alt = self.edges_alt[:]
        #self.draw_edges_alt.insert(0,0)
        #print self.edges
        #for i in range(len(self.edges)):
        #    if self.edges[i]:
        #        print i

    def print_path(self):
        """Prints the path edges according to test_graph's draw grids method."""

        grid = tg.Graph.grid_graph(self.graph.rows,self.graph.cols)
        #tg.draw_grid(self.draw_edges_alt,self.graph.rows,self.graph.cols,grid)
        tg.draw_grid(self.draw_edges,self.graph.rows,self.graph.cols,grid)
        


    def find_path(self):
        """Finds the line number for the path.

        Finds the first line of data for that path in the gps file.
        Sets the line_num if it is not already set.

        Returns:
            The line num of the first line for that path entry in the gps file.
        """
        
        if self.line_num != -1:
            return self.line_num

        max_line = self.graph.gps_length - 1
        min_line = 0
        last_id = dg.normalize(self.graph.lines[-1])[0]
        pivot = int((self.trip_id-1)/float(last_id)*self.graph.gps_length)
        cur_id = dg.normalize(self.graph.lines[pivot])[0]
        while cur_id != self.trip_id:
            if cur_id < self.trip_id:
                min_line = pivot
            else:
                max_line = pivot
            #TODO: could make this run in essentially constant time by hopping predetermined distance
            pivot = (min_line + max_line) / 2
            cur_id = dg.normalize(self.graph.lines[pivot])[0]

        while  dg.normalize(self.graph.lines[pivot])[0] == self.trip_id:
            pivot -= 1

        pivot += 1
        self.line_num = pivot
        return pivot


    #@profile
    def create_path(self):
        """Creates the path grid for the path's trip_id.

        It creates an array to model the graph's dimensions.
        Array spots are 1 if the path traverses them.
        Array spots are 0 if path doesnt traverse them.
        If a coordinate is outside the graph, do not include it in the graph.
        Finds the longest collection of points in between exits from the graph.
        Updates graph node visit information by calling graph.node_visit().
        Sets self.next_line value.
        
        TODO: implement this
        Mark a path as bad if it is not a legal path (if two nodes are visited
        sequentially, but are not adjacent).

        Returns:
            A numpy array with dimensions equal to those of the input graph.
            Grid spots are 1 if the path traverses them, 0 otherwise.

            The set of edges corresponding to that path.

            A boolean that is true if all adjacent points have valid edges, false otherwise.
        """

        matrices = []
        matrices.append([np.zeros((self.graph.rows,self.graph.cols)),0])
        edge_sets = []
        edge_sets.append([0 for i in range(self.graph.num_edges)])
        cur_line = self.line_num
        good_graphs = []
        good_graphs.append(True)
        nodes_visited = []
        nodes_visited.append([])
        normalized = dg.normalize(self.graph.lines[cur_line])
        matrices_index = 0
        prev_coords = (-1,-1)
        while normalized[0] == self.trip_id:
            lat = normalized[1]
            lon = normalized[2]
            coords = self.graph.gps_to_coords(lat,lon)

            if prev_coords != (-1,-1) and coords[0] != -1 and coords != prev_coords:
                edge_num = self.graph.edge_num(prev_coords[0],prev_coords[1],coords[0],coords[1])
                if edge_num == -1:
                    good_graphs[matrices_index] = False
                edge_sets[matrices_index][edge_num] = 1

            if coords[0] == -1:
                matrices.append([np.zeros((self.graph.rows,self.graph.cols)),0])
                edge_sets.append([0 for i in range(self.graph.num_edges)])
                good_graphs.append(True)
                nodes_visited.append([])
                matrices_index += 1
            
            elif coords[0] < self.graph.rows and coords[1] < self.graph.cols and not matrices[matrices_index][0][coords[0]][coords[1]]:
                matrices[matrices_index][1] += 1
                matrices[matrices_index][0][coords[0]][coords[1]] = 1
                nodes_visited[matrices_index].append(coords)

            prev_coords = coords

            cur_line += 1
            if cur_line == len(self.graph.lines):
                break
            normalized = dg.normalize(self.graph.lines[cur_line])

        self.next_line = cur_line
        best_index = 0
        best_score = 0
        for matrix_index in range(len(matrices)):
            if matrices[matrix_index][1] > best_score:
                best_score = matrices[matrix_index][1]
                best_index = matrix_index

        for coords in nodes_visited[best_index]:
            self.graph.node_visit(self.trip_id,coords)

        return matrices[best_index][0],edge_sets[best_index],good_graphs[best_index]



    def path_to_edges(self):
        """Maps the node path to edges used.

        DON'T USE THIS: IT DOESN'T TAKE INTO ACCOUNT PATH ORDERING

        Determines which edges are used in the node path.
        Returned array is 0-indexed, but the sdd interpretation has this 1-indexed.

        This representation
            X 0 X 2 X
            1   3   6
            X 4 X 7 X
            5   8  10
            X 9 X 11X

        Sdd representation
            X 1 X 3 X
            2   4   7
            X 5 X 8 X
            6   9  11
            X 10X 12X


        Returns:
            An array of length cols*(rows-1) + rows*(cols-1).
            Each entry is 1 if the corresponding edge is used, 0 otherwise.

        """

        edges = [0 for i in range(self.graph.num_edges)]

        for row in range(self.graph.rows):
            for col in range(self.graph.cols):
                if self.path[row][col]:
                    if row + col < self.graph.cols - 1:
                        if col < self.graph.cols - 1 and self.path[row][col + 1]:
                            #print "(%d,%d) (%d,%d)" % (row, col, row, col + 1)
                            edge_number = self.graph.diags[row + col] + 2 * row
                            edges[edge_number] = 1
                        if row < self.graph.rows - 1 and self.path[row + 1][col]:
                            #print "(%d,%d) (%d,%d)" % (row, col, row + 1, col)
                            edge_number = self.graph.diags[row + col] + 1 + 2 * row
                            edges[edge_number] = 1
                    else:
                        col_dist = self.graph.cols - col - 1
                        if col < self.graph.cols - 1 and self.path[row][col + 1]:
                            #print "(%d,%d) (%d,%d)" % (row, col, row, col + 1)
                            edge_number = self.graph.diags[row + col] + 2 * col_dist  - 1
                            edges[edge_number] = 1
                        if row < self.graph.rows - 1 and self.path[row + 1][col]:
                            #print "(%d,%d) (%d,%d)" % (row, col, row + 1, col)
                            edge_number = self.graph.diags[row + col] + 2 * col_dist
                            edges[edge_number] = 1
                        

        return edges

#@profile
def create_all(graph):
    """Creates a dict containing every path in the file.

    Used to determine the best node and paths through that node.

    Attributes:
        graph: the graph.

    Returns:
        dict of paths, indexed by trip_id
    """
    full_fn = open('csvGPS.txt','r')
    lines = full_fn.readlines()
    file_length = len(lines)
    full_fn.close()
    full_fn = open('csvGPS.txt','r')
    trip_id = 1
    line_num = 0
    #paths = {}
    p = Path(trip_id,graph,full_fn,line_num)
    full_fn.close()
    #paths[trip_id] = p
    while p.next_line != file_length:
        graph.trip_id2line_num[trip_id] = line_num
        full_fn = open('csvGPS.txt','r')
        line_num = p.next_line
        trip_id = dg.normalize(lines[line_num])[0]
        p = Path(trip_id,graph,full_fn,line_num)
        full_fn.close()
       # paths[trip_id] = p
    #return paths
        




def main():
    full_fn = open('csvGPS.txt','r')
    orig_fn = open('firstLast.txt','r')

    """full SF coords
    min_lat = 37.72
    max_lat = 37.808
    min_lon = -122.515
    max_lon = -122.38
    """

    """SF action coords (3.8x3.3 mi)"""
    min_lat = 37.755
    max_lat = 37.803
    min_lon = -122.46
    max_lon = -122.39

    rows = 10 
    cols = 10
    g = Graph(full_fn,min_lat,max_lat,min_lon,max_lon,rows,cols)
    try_lat = 37.721396 
    try_lon = -122.400256


    """
    full_fn = open('csvGPS.txt','r')
    p = Path(1,g,full_fn,0)
    full_fn.close()
    full_fn = open('csvGPS.txt','r')
    lines = full_fn.readlines()
    print lines[0]
    print lines[27]
    full_fn.close()

    for i in (20,21,22,23,24,25,26,27,28):
        full_fn = open('csvGPS.txt','r')
        p = Path(i,g,full_fn)
        full_fn.close()
    
    print g.node2visited
    print g.node2trip_ids
    print g.best_node_score
    print g.best_node
    """
    create_all(g)
    print g.best_node_score
    print g.best_node
    print g.node_to_coords(g.best_node)
    
    trip_list = g.node2trip_ids[g.best_node]
    for i in range(220,230):
        trip_id = trip_list[i]
        line_num = g.trip_id2line_num[trip_id]
        p = Path(trip_id,g,line_num)
        print trip_id
        p.print_path()
        

    """ Create Epochs of Data 
    trip_list = g.node2trip_ids[g.best_node]
    for i in range(10):
        filename = "uber-data_%d_%d_%d.txt" % (rows,cols,i)
        fn = open(filename,"w")
        fn.close()

    for i in range(len(trip_list)):
        trip_id = trip_list[i]
        #print trip_id
        line_num = g.trip_id2line_num[trip_id]
        p = Path(trip_id,g,line_num)
        epoch = int(float(i) / len(trip_list) * 10)
        print epoch
        filename = "uber-data_%d_%d_%d.txt" % (rows,cols,epoch)
        fn = open(filename,"a")
        fn.write(str(p.edges)[1:-1])
        fn.write("\n")
        fn.close()
        
        #print p.trip_id
        #print p.path
        #p.print_path()
    """

    return

    trips = dg.createTrips(orig_fn)

    full_trips,best_coords,trips_with_point = dg.create_full(full_fn,trips,
                                                            g.lat_step,
                                                            g.lon_step,
                                                            g.min_lat,
                                                            g.min_lon)
    print best_coords
    print len(trips_with_point)
    coords =  g.gps_to_coords(best_coords[0],best_coords[1])
    node_num = g.coords_to_node(coords[0],coords[1])
    print coords
    print node_num

    full_fn.close()
    full_fn = open('csvGPS.txt','r')
    #print g.gps_to_coords(try_lat,try_lon)

    for i in range(10,25):
        full_fn.close()
        full_fn = open('csvGPS.txt','r')
        p = Path(trips_with_point[i],g,full_fn)
        p.print_path()

    #trip_id = int(sys.argv[1])
    #p = Path(trip_id,g,full_fn)
    
if __name__ == '__main__':
    main()
