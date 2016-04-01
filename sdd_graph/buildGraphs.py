#!/usr/bin/python
import sys
import random
import pickle
import heapq
import math
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

        edge_filename = 'graphs/edge-nums-%d-%d.pickle' % (self.rows,self.cols)
        self.edge2index = pickle.load(open(edge_filename,'rb'))

        self.lat_step = float((max_lat-min_lat)/rows)
        self.lon_step = float((max_lon-min_lon)/cols)

        #self.diags = self.create_diags()
        self.num_edges = self.rows*(self.cols-1) + self.cols*(self.rows-1)  #self.diags[self.rows+self.cols-2]
        print "num edges: %d" % self.num_edges
        self.node2visited = defaultdict(int) 
        self.node2trip_ids = defaultdict(list)
        self.best_node = 1
        self.best_node_score = 0
        #self.trip_id2lengths,self.avg_length = self.path_lengths()

    def path_lengths(self):
        """Determines the length of each path in the dataset

        Returns:
            A dict mapping trip_id (key) to path length (value)
            The average path length in the dataset
        """
        trip_id2length = defaultdict(float)
        prev_id = 0
        cur_id = 0
        prev_lat = 0
        prev_lon = 0
        big_hops = {}
        print "Bad Distances"
        for line in self.lines:
            #normalized = dg.normalize(line)
            normalized = normalize_simple(line)
            cur_id = normalized[0]
            lat = normalized[1]
            lon = normalized[2]
            if cur_id == prev_id:
                distance = gps_dist_miles(prev_lat,prev_lon,lat,lon)
                if distance > 1:
                    big_hops[cur_id] = 1
                    #print cur_id
                trip_id2length[cur_id] += distance 
            prev_lat = lat
            prev_lon = lon
            prev_id = cur_id

        print len(trip_id2length.keys())
        for bad_id in big_hops.keys():
            del trip_id2length[bad_id]

        for i in (15,18,333,24,12345):
            print "%d: %f" % (i,trip_id2length[i])

        #for i in range(1,25001):
        #    if i not in trip_id2length.keys():
        #        print i
        num_trips = len(trip_id2length.keys())
        print num_trips
        total_len = 0.0
        for i in range(1,25001):
            if i not in trip_id2length.keys():
                continue
            if trip_id2length[i] > 50:
                del trip_id2length[i]
            total_len += trip_id2length[i]
        heap = []
        for i in trip_id2length.keys():
            heapq.heappush(heap,trip_id2length[i])
        quarter_len = num_trips/4
        for i in range(quarter_len):
            heapq.heappop(heap)
        print "25th percentile: %f" % heapq.heappop(heap)
        for i in range(quarter_len):
            heapq.heappop(heap)
        print "median: %f" % heapq.heappop(heap)
        for i in range(quarter_len):
            heapq.heappop(heap)
        print "75th percentile: %f" % heapq.heappop(heap)

        num_trips = len(trip_id2length.keys())
        print num_trips
        avg_len = total_len/num_trips
        print "average length: %f" % avg_len 
        print "total length %f" % total_len
        return trip_id2length,avg_len


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


    def top_n_nodes(self,num_best):
        heap = []
        for node_num in self.node2visited.keys():
            neg_num_visits = 0 - self.node2visited[node_num]
            heapq.heappush(heap,(neg_num_visits,node_num))
        best_nodes = []
        for i in range(num_best):
            num_visits,node_num = heapq.heappop(heap)
            num_visits = 0 - num_visits
            best_nodes.append(node_num)
            print node_num
            print self.node_to_coords(node_num)
            print num_visits
            print ""
        return best_nodes


    def edge_num(self,row1,col1,row2,col2):
        """Determines the edge number between two points.

        THIS METHOD DOES NOT WORK WITH ACTUAL GRAPH ORDERING BECAUSE GRAPHILLION HAS A BUG AND DOESN'T ORDER EDGES LOGICALLY

        The method determines which point comes first.

        These numberings are not correct because of graphillion bug
        vvvvvvvvvvvvvvvvvvvvvvvv

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
        
        if not ((row == row_n and col == col_n - 1) or (row == row_n-1 and col == col_n)):
            return -1

        if row < 0 or row_n >= self.rows or col < 0 or col_n >= self.cols:
            return -1
        
        node1 = row*self.rows+col+1
        node2 = row_n*self.rows+col_n+1
        edge_number = self.edge2index[(node1,node2)]
        #print "%s %s: %d" % (str(node1),str(node2),edge_number)
        """
        #THIS DOWN HERE WOULD WORK IF GRAPHILLION NUMBERED EDGES CORRECTLY BUT IT DOESNT
        #print "(%d,%d) (%d,%d)" % (row,col,row_n,col_n)
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
        """

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

        #print diag_counts
        #print diag_full
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

        if (lat <= self.min_lat or lat >= self.max_lat or lon <= self.min_lon or lon >= self.max_lon):
            return (-1,-1)

        lat_spot = int((self.max_lat-lat)/self.lat_step)
        lon_spot = int((lon-self.min_lon)/self.lon_step)
        #print "lat: %f lon: %f lat_spot: %f lon_spot: %f" % (lat,lon,lat_spot,lon_spot)
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
    def __init__(self,trip_id,graph,line_num=-1,midpoint=None):
        self.graph = graph
        self.trip_id = trip_id
        self.line_num = line_num
        self.bad_graph = False
        self.next_line = 0
        self.line_num = self.find_path()
        self.midpoint = midpoint
        if midpoint == None:
            self.midpoint = self.graph.best_node
        self.path,self.edges,self.good,self.partials = self.create_path()

    def print_path(self):
        """Prints the path edges according to test_graph's draw grids method."""

        grid = tg.Graph.grid_graph(self.graph.rows,self.graph.cols)
        #tg.draw_grid(self.draw_edges_alt,self.graph.rows,self.graph.cols,grid)
        tg.draw_grid(self.edges,self.graph.rows,self.graph.cols,grid)
        


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
        #last_id = dg.normalize(self.graph.lines[-1])[0]
        last_id = normalize_simple(self.graph.lines[-1])[0]
        pivot = int((self.trip_id-1)/float(last_id)*self.graph.gps_length)
        #cur_id = dg.normalize(self.graph.lines[pivot])[0]
        cur_id = normalize_simple(self.graph.lines[pivot])[0]
        while cur_id != self.trip_id:
            if cur_id < self.trip_id:
                min_line = pivot
            else:
                max_line = pivot
            #TODO: could make this run in essentially constant time by hopping predetermined distance
            pivot = (min_line + max_line) / 2
            #cur_id = dg.normalize(self.graph.lines[pivot])[0]
            cur_id = normalize_simple(self.graph.lines[pivot])[0]

        #while dg.normalize(self.graph.lines[pivot])[0] == self.trip_id:
        while normalize_simple(self.graph.lines[pivot])[0] == self.trip_id:
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
        

        Returns:
            A numpy array with dimensions equal to those of the input graph.
            Grid spots are 1 if the path traverses them, 0 otherwise.

            The set of edges corresponding to that path.
                Each edge is 0 if it is not included and 1 if it is included.

            A boolean that is true if all adjacent points have valid edges, false otherwise.

            A dict that contains the partial path from the start of the path to the midpoint.
                Edges from the start to the midpoint are set to 1.
                Edges around the start point and not used in the path are set to 0.
                    This is the entirity of the information we know about the final path from the partial.
                    We know that the path is not any longer in the direction away from the midpoint.
        """

        partials = []
        partials.append({})
        print self.trip_id

        #this variable is true if we have not yet recorded the first edge of a path
        first_edge = True
        #this variable is false until we hit the midpoint
        hit_midpoint = False

        matrices = []
        matrices.append([np.zeros((self.graph.rows,self.graph.cols)),0])
        edge_sets = []
        edge_sets.append([0 for i in range(self.graph.num_edges)])
        cur_line = self.line_num
        good_graphs = []
        good_graphs.append(True)
        nodes_visited = []
        nodes_visited.append([])
        #normalized = dg.normalize(self.graph.lines[cur_line])
        normalized = normalize_simple(self.graph.lines[cur_line])
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
                else:
                    edge_sets[matrices_index][edge_num] = 1
                    if edge_num in partials[matrices_index] and partials[matrices_index][edge_num] == 0:
                        del partials[matrices_index][edge_num]
                    if not hit_midpoint:
                        if first_edge:
                            above = (prev_coords[0]-1,prev_coords[1])
                            below = (prev_coords[0]+1,prev_coords[1])
                            left = (prev_coords[0],prev_coords[1]-1)
                            right = (prev_coords[0],prev_coords[1]+1)
                            for next_coords in (above,below,left,right):
                                other_edge = self.graph.edge_num(prev_coords[0],prev_coords[1],next_coords[0],next_coords[1])
                                if other_edge != -1:
                                    partials[matrices_index][other_edge] = 0
                            first_edge = False
                            if self.graph.coords_to_node(prev_coords[0],prev_coords[1]) == self.midpoint:
                                hit_midpoint = True
                        partials[matrices_index][edge_num] = 1
                        if self.graph.coords_to_node(coords[0],coords[1]) == self.midpoint:
                            hit_midpoint = True



            if coords[0] == -1:
                matrices.append([np.zeros((self.graph.rows,self.graph.cols)),0])
                edge_sets.append([0 for i in range(self.graph.num_edges)])
                good_graphs.append(True)
                nodes_visited.append([])
                matrices_index += 1
                partials.append({})
                hit_midpoint = False
                first_edge = True
            
            elif coords[0] < self.graph.rows and coords[1] < self.graph.cols and not matrices[matrices_index][0][coords[0]][coords[1]]:
                matrices[matrices_index][1] += 1
                matrices[matrices_index][0][coords[0]][coords[1]] = 1
                nodes_visited[matrices_index].append(coords)

            prev_coords = coords

            cur_line += 1
            if cur_line == len(self.graph.lines):
                break
            #normalized = dg.normalize(self.graph.lines[cur_line])
            normalized = normalize_simple(self.graph.lines[cur_line])

        self.next_line = cur_line
        best_index = 0
        best_score = 0
        for matrix_index in range(len(matrices)):
            if matrices[matrix_index][1] > best_score:
                best_score = matrices[matrix_index][1]
                best_index = matrix_index

        for coords in nodes_visited[best_index]:
            self.graph.node_visit(self.trip_id,coords)

        return matrices[best_index][0],edge_sets[best_index],good_graphs[best_index],partials[best_index]



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

def find_next_comma_newline(line,index):
    """Find the index of the next comma or newline character in a line.

    Attributes:
        string: line
        int: index 
            The position from which you want to find the next of those characters.
            First check the next position

    Returns:
        int: the next occurrence of a comma or newline
            return -1 if there is no other comma or newline
    """

    index += 1
    while index < len(line) and line[index] != "," and line[index] != "\n":
        index += 1
    if index == len(line):
        return -1
    return index

def normalize_simple(line):
    """Finds trip id, latitude and longitude of a line.

    Lines must be of the form "id,latitude,longitude,..."
        or of the form "id,latitude,longitude\n"

    Returns:
        int: trip id
        float: latitude
        flot: longitude
    """
    first = find_next_comma_newline(line,0)
    #print "first: %d" % first
    second = find_next_comma_newline(line,first+1)
    #print "second: %d" % second
    third = find_next_comma_newline(line,second+1)
    #print "third: %d" % third
    if third == -1:
        lon = float(line[second+1:])
    else:
        lon = float(line[second+1:third])
    return int(line[0:first]),float(line[first+1:second]),lon



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
    trip_id = 1
    line_num = 0
    #paths = {}
    p = Path(trip_id,graph,line_num=line_num)
    #paths[trip_id] = p
    while p.next_line != file_length:
        graph.trip_id2line_num[trip_id] = line_num
        line_num = p.next_line
        trip_id = normalize_simple(lines[line_num])[0]
        #trip_id = dg.normalize(lines[line_num])[0]
        p = Path(trip_id,graph,line_num=line_num)
       # paths[trip_id] = p
    graph.trip_id2line_num[trip_id] = line_num
    #return paths
        
def create_all_cab(graph):
    """Creates a dict containing every path in the file"""
    return

def single_epoch(g,rows,cols):
    """Create a single epoch of data.

    Outputs a text file with all of the edge instatniations for all trips that pass through the best point.

    Returns:
        Nothing
    """

    num_top = 5
    #3 for 8x8
    one_to_select = 0 
    top_nodes = g.top_n_nodes(num_top)
    '''
    for k in range(num_top):
        node_num = top_nodes[k]
        trip_list = g.node2trip_ids[node_num]
        print "Next Midpoint: %d" % k
        print node_num
        print g.node_to_coords(node_num)
        print "Num trips: %d" % len(trip_list)
        for i in range(len(trip_list)):
            trip_id = trip_list[i]
            line_num = g.trip_id2line_num[trip_id]
            p = Path(trip_id,g,line_num)
            """
            print i
            print trip_id
            p.print_path()
            for i in range(p.graph.num_edges):
                if p.edges[i]:
                    sys.stdout.write("%d, " % (i + 1))
            sys.stdout.write("\n")
            sys.stdout.write("1s: ")
            for key in p.partials.keys():
                if p.partials[key]:
                    sys.stdout.write("%d, " % (key + 1))
            sys.stdout.write("\n0s: ")
            for key in p.partials.keys():
                if not p.partials[key]:
                    sys.stdout.write("%d, " % (key + 1))
            sys.stdout.write("\n")
            #"""
    '''

    #trip_list = g.node2trip_ids[g.best_node]
    midpoint = top_nodes[one_to_select]
    trip_list = g.node2trip_ids[midpoint]
    print "Selected midpoint: %d" % top_nodes[one_to_select]
    print g.node_to_coords(midpoint)
    out_file = open("datasets/full_data_%d_%d.txt" % (rows,cols),'w')
    partial_file = open("datasets/partials_%d_%d.txt" % (rows,cols), 'w')
    for i in range(len(trip_list)):
        trip_id = trip_list[i]
        line_num = g.trip_id2line_num[trip_id]
        p = Path(trip_id,g,line_num=line_num,midpoint=midpoint)
        """
        print i
        print trip_id
        p.print_path()
        for i in range(p.graph.num_edges):
            if p.edges[i]:
                sys.stdout.write("%d, " % (i + 1))
        sys.stdout.write("\n")
        sys.stdout.write("1s: ")
        for key in p.partials.keys():
            if p.partials[key]:
                sys.stdout.write("%d, " % (key + 1))
        sys.stdout.write("\n0s: ")
        for key in p.partials.keys():
            if not p.partials[key]:
                sys.stdout.write("%d, " % (key + 1))
        sys.stdout.write("\n")
        """
        out_string = str(p.edges)[1:-1]
        out_file.write("%s\n" % out_string)
        for i in range(p.graph.num_edges):
            if i in p.partials.keys():
                partial_file.write("%d" % p.partials[i])
            else:
                partial_file.write("-1")
            if i < p.graph.num_edges-1:
                partial_file.write(",")
        partial_file.write("\n")

    out_file.close()

def create_epochs(g,rows,cols):
    """ Create Epochs of Data """
    trip_list = g.node2trip_ids[g.best_node]
    for i in range(10):
        filename = "datasets/uber-data_%d_%d_%d.txt" % (rows,cols,i)
        fn = open(filename,"w")
        fn.close()

    for i in range(len(trip_list)):
        trip_id = trip_list[i]
        #print trip_id
        line_num = g.trip_id2line_num[trip_id]
        p = Path(trip_id,g,line_num=line_num)
        epoch = int(float(i) / len(trip_list) * 10)
        #print epoch
        out_string = str(p.edges)[1:-1]
        """
        if out_string[-1] == "1":
            print trip_id
            print p.edges
            print p.path
        """
        rand_num = random.random()*10
        """
        if rand_num >= 9:
            print "random: %d" % trip_id
            print p.edges
            print p.path
        """

        filename = "datasets/uber-data_%d_%d_%d.txt" % (rows,cols,epoch)
        fn = open(filename,"a")
        fn.write(str(p.edges)[1:-1])
        fn.write("\n")
        fn.close()
        
        #print p.trip_id
        #print p.path
        #p.print_path()
    
def print_some(g,trip_nums):
    trip_list = g.node2trip_ids[g.best_node]
    for i in trip_nums:
        trip_id = trip_list[i]
        line_num = g.trip_id2line_num[trip_id]
        p = Path(trip_id,g,line_num=line_num)
        print trip_id
        print p.edges
        print p.path
        p.print_path()


def gps_dist_miles(lat1, long1, lat2, long2):
    if lat1 == lat2 and long1 == long2:
        return 0.0
 
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
         
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
         
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
     
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
 
    earth_radius_miles = 3959

    return arc*earth_radius_miles

def main():

    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    #full_fn = open('csvGPS.txt','r')
    full_fn = open('cab_trips.txt','r')
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
   
    """SF zoom coords (2.2x2.4 mi)
    min_lat = 37.763
    max_lat = 37.795
    min_lon = -122.445
    max_lon = -122.4
    """

    g = Graph(full_fn,min_lat,max_lat,min_lon,max_lon,rows,cols)

    create_all(g)
    print g.best_node_score
    print g.best_node
    print g.node_to_coords(g.best_node)
  
    fourBad = (109,553,416,194,558,629,179,216)
    fiveBad = (702,203,20,570,491)
    sixBad = (353,105,476,455,166+482,5+482,84+482)
    sevenBad = (291,229,243,363,345)
    tenBad = (133,256,10,278,203,166,221,177,191,115,297,281,78,89,180)


    #print_some(g,fiveBad)

    single_epoch(g,rows,cols)

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

    
if __name__ == '__main__':
    main()
