#!/usr/bin/python
import sys
import pickle
import buildGraphs as bg
from collections import defaultdict

def normalize(line):
    """Taken from cabspottingdata/to_trips.py"""
    counter = 0
    while line[counter] != ',':
        counter += 1
    first_comma = counter
    lat = float(line[:first_comma])
    counter += 1
    while line[counter] != ',':
        counter += 1
    second_comma = counter
    lon = float(line[first_comma+1:second_comma])
    counter += 2
    occupied = int(line[counter-1])
    time = int(line[counter+1:])
    return lat,lon,occupied,time

def gps_to_coords(lat,lon,min_lat,max_lat,min_lon,max_lon,lat_step,lon_step):
    """Taken from buildGraphs.py
    Determines the coodinates on the graph corresponding to a given gps point.

    Attributes:
        lat: the latitude of the point
        lon: the longitude of the point

    Returns:
        A pair corresponding to the (row,column) in the graph that holds that gps point
        (-1,-1) if the coordinates are out of scope
    """

    if (lat <= min_lat or lat >= max_lat or lon <= min_lon or lon >= max_lon):
        return (-1,-1)

    lat_spot = int((max_lat-lat)/lat_step)
    lon_spot = int((lon-min_lon)/lon_step)
    #print "lat: %f lon: %f lat_spot: %f lon_spot: %f" % (lat,lon,lat_spot,lon_spot)
    return (lat_spot,lon_spot)

def coords_to_node(cols,row,col):
    """Taken from buildGraphs.py
    determines the node index of a coordinate in the path graph

    1 2 3
    4 5 6
    7 8 9

    Attributes:
        row: row number (0 indexed)
        col: col number (0 indexed)

    Returns:
        node number: (1 indexed)
    """
    return row*cols + col + 1

def main():

    rows = int(sys.argv[1])
    cols = int(sys.argv[2])

    """SF action coords (3.8x3.8 mi) - 95% of all paths pass through here"""
    min_lat = 37.75
    max_lat = 37.806
    min_lon = -122.46
    max_lon = -122.39
    lat_step = float((max_lat-min_lat)/rows)
    lon_step = float((max_lon-min_lon)/cols)

    ids = pickle.load(open('../cabspottingdata/taxi_ids.pickle','rb'))

    node2total_lon = defaultdict(float)
    node2total_lat = defaultdict(float)
    node2count = defaultdict(int)

    for trip_id in ids:
        fn = open('../cabspottingdata/csv_%s.txt' % trip_id,'r')
        lines = fn.readlines()
        fn.close()
        for line in lines:
            lat,lon,occ,time = normalize(line)
            row,col = gps_to_coords(lat,lon,min_lat,max_lat,min_lon,max_lon,lat_step,lon_step)
            node = coords_to_node(cols,row,col)
            node2count[node] += 1
            node2total_lat[node] += lat
            node2total_lon[node] += lon

    node2avg_gps = {}
    for node in node2count:
        node_ct = node2count[node]
        node2avg_gps[node] = (node2total_lat[node]/node_ct,node2total_lon[node]/node_ct)

    with open('pickles/node2avg_gps.pickle','wb') as output:
        pickle.dump(node2avg_gps,output)




if __name__ == '__main__':
    main()
