#!/usr/bin/python
import pickle
import datetime
import buildGraphs as bg
from collections import defaultdict
import heapq

def get_unix_time(line):
    i = len(line)-1
    while line[i] != ',':
        i -= 1
    i += 1
    return int(line[i:])

def convert_unix_time(unix):
    """return [day_of_week (MONDAY is 0),hour,minute]
    
    This function only works for dates that were in may/june of 2008
    """
    date = datetime.datetime.fromtimestamp(unix)
    day = date.day + (date.month-5)*31
    day_of_week = day % 7 - 5
    if day_of_week < 0:
        day_of_week += 7
    print "day of week: %d" % day_of_week
    return [day_of_week,date.hour,date.minute]


def create_mappings(in_filename,out_filename):
    """Create pickled dict mapping from trip_id to trip start time

    Store time as [day of the week,start hour,start minute]
    """

    with open(in_filename,'r') as infile:
        lines = infile.readlines()

    trip_id2line_num = pickle.load(open('trip_id2line_num.pickle','rb'))

    trip_id2time_obj = {}

    for i in trip_id2line_num.keys():
        line_num = trip_id2line_num[i]
        line = lines[line_num]
        unix = get_unix_time(line)
        time_obj = convert_unix_time(unix)
        trip_id2time_obj[i] = time_obj

    print len(trip_id2time_obj)

    with open(out_filename,'wb') as output:
        pickle.dump(trip_id2time_obj,output)


def analyze_times(to_time_fn):
    trip_id2time = pickle.load(open(to_time_fn,'rb'))

    day_hour2trip_ids = defaultdict(list)

    for trip_id in trip_id2time.keys():
        time_obj = trip_id2time[trip_id]
        day_hour2trip_ids[(time_obj[0],time_obj[1])].append(trip_id)

    sorted_times = []
    for time in day_hour2trip_ids.keys():
        counts = day_hour2trip_ids[time]
        heapq.heappush(sorted_times,[(0-counts),time])
        #print "%s: %d" % (str(time),len(day_hour2trip_ids[time]))

    while len(sorted_times) > 0:
        popped = heapq.heappop(sorted_times)
        print "%s: %d" % (str(popped[1]),(0-popped[0]))



def main():
    to_time_filename = 'trip_id2time.pickle'
    #create_mappings('cab_chronological.txt',to_time_filename)
    analyze_times(to_time_filename)




if __name__ == '__main__':
    main()
