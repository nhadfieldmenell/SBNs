#!/usr/bin/python
import pickle
import datetime
import buildGraphs as bg

def get_unix_time(line):
    i = len(line)-1
    while line[i] != ',':
        i -= 1
    i += 1
    return int(line[i:])

def convert_unix_time(unix):
    """return [day_of_week (Monday is 0),hour,minute]"""
    date = datetime.datetime.fromtimestamp(unix)
    print "date object: %s" % str(date)
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

    for i in (5,77,15,1235,1452):
        line_num = trip_id2line_num[i]
        line = lines[line_num]
        unix = get_unix_time(line)
        time_obj = convert_unix_time(unix)
        print time_obj

    #print get_unix_time(lines[trip_id2line_num[66]])

def main():
    create_mappings('cab_chronological.txt','trip_id2time.pickle')


if __name__ == '__main__':
    main()
