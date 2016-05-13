#!/usr/bin/python
import sys
import pickle
import heapq
import buildGraphs as bg
import create_node2avg_gps as c2a
from collections import defaultdict

def main():
    node2heap = defaultdict(list)
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

    count = 0
    for cab_id in ids:
        print count
        count += 1
        fn = open('../cabspottingdata/csv_%s.txt' % cab_id,'r')
        lines = fn.readlines()
        fn.close()
        for line in lines:
            lat,lon,occ,time = c2a.normalize(line)
            row,col = c2a.gps_to_coords(lat,lon,min_lat,max_lat,min_lon,max_lon,lat_step,lon_step)
            node = c2a.coords_to_node(cols,row,col)
            heapq.heappush(node2heap[node],[lat,lon])

    node2median = {}
    for node in node2heap:
        heap = node2heap[node]
        for i in range(len(heap)/2):
            heapq.heappop(heap)
        median = heapq.heappop(heap)
        node2median[node] = median

    with open('pickles/node2median_%d_%d.pickle' % (rows,cols),'wb') as output:
        pickle.dump(node2median,output)

if __name__ == '__main__':
    main()
