#!/usr/bin/python
import pickle
import sys

def main():
    row = int(sys.argv[1])
    col = int(sys.argv[2])
    first_last2trip_ids = pickle.load(open('pickles/first_last2trip_ids-%d-%d.pickle' % (row,col),'rb'))
    trip_id2first_last = {}
    for first_last in first_last2trip_ids:
        trips = first_last2trip_ids[first_last]
        for trip in trips:
            trip_id2first_last[trip] = first_last

    with open('pickles/trip_id2first_last-%d-%d.pickle' % (row,col),'wb') as output:
        pickle.dump(trip_id2first_last,output)

if __name__ == '__main__':
    main()
