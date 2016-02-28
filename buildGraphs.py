#!/usr/bin/python

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

    def __init__(self,min_lat,max_lat,min_lon,max_lon,rows,cols):
        self.rows = rows
        self.cols = cols
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon

        self.lat_step = float((max_lat-min_lat)/rows)
        self.lon_step = float((max_lon-min_lon)/cols)


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

        lat_spot = int((lat-self.min_lat)/self.lat_step)
        lon_spot = int((lon-self.min_lon)/self.lon_step)
        print self.min_lon
        print lon
        return (lat_spot,lon_spot)

def main():
    min_lat = 37.72
    max_lat = 37.808
    min_lon = -122.515
    max_lon = -122.38
    rows = 10
    cols = 10
    g = Graph(min_lat,max_lat,min_lon,max_lon,rows,cols)
    try_lat = 37.735760
    try_lon = -122.448849 
    print g.gps_to_coords(try_lat,try_lon)
    
    
if __name__ == '__main__':
    main()
