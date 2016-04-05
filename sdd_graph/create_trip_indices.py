#!/usr/bin/python

from buildGraphs import normalize_simple as normalize
import pickle

def main():
    """Create a pickled dict which maps trip_num to first line that trip appears on in cabs_chronological.txt"""

    trip_id2line_num = {}

    cab_trips = open('cab_chronological.txt','r')
    lines = cab_trips.readlines()
    cab_trips.close()
    prev_id = 0

    for i in range(len(lines)):
        cur_id = normalize(lines[i])[0]
        if cur_id != prev_id:
            print cur_id
            trip_id2line_num[cur_id] = i
        prev_id = cur_id

    #print trip_id2line_num.keys()

    with open('trip_id2line_num.pickle','wb') as output:
        pickle.dump(trip_id2line_num,output)


if __name__ == '__main__':
    main()
