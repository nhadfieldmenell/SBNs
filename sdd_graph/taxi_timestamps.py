#!/usr/bin/python
import pickle
import buildGraphs as bg

def get_unix_time(line):
    i = len(line)-1
    while i != ',':
        i -= 1
    i += 1
    return int(line[i:])

def create_mappings(in_filename,out_filename):
    """Create pickled dict mapping from trip_id to trip start time

    Store time as [day of the week,start hour,start minute]
    """

    with open(in_filename,'r') as infile:
        lines = infile.readlines()

    trip_id2line_num = pickle.load(open('trip_id2line_num.pickle','rb'))

    print get_unix_time(lines[trip_id2line_num[66]])

def main():
    create_mappings('cab_chronological.txt','trip_id2time.pickle')


if __name__ == '__main__':
    main()
