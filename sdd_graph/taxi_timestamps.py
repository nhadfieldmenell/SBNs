#!/usr/bin/python
import pickle
import datetime
import buildGraphs as bg
from collections import defaultdict
import heapq

def get_unix_time(line):
    """Parse unix timestamp from a line of data in taxi dataset
    """
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


def analyze_times(trip_id2time):
    """Prints out mapping of (day,hour) to number of trips in dataset that depart at that time

    Elements printed in descending order of number of trips per element
    """

    day_hour2trip_ids = defaultdict(list)

    for trip_id in trip_id2time.keys():
        time_obj = trip_id2time[trip_id]
        day_hour2trip_ids[(time_obj[0],time_obj[1])].append(trip_id)

    sorted_times = []
    for time in day_hour2trip_ids.keys():
        counts = 0-len(day_hour2trip_ids[time])
        heapq.heappush(sorted_times,[counts,time])
        #print "%s: %d" % (str(time),len(day_hour2trip_ids[time]))

    while len(sorted_times) > 0:
        popped = heapq.heappop(sorted_times)
        print "%s: %d" % (str(popped[1]),(0-popped[0]))


def classify_trips(trip_id2time):
    """Creates a mapping of trip id to time class of that trip

    Classes:
        0 - Weekend Evening:
            Friday 7PM-Midnight
            Saturday Midnight-7AM
            Saturday 7PM-Midnight
            Sunday Midnight-7AM
            
        1 - Weekend Day:
            Saturday 7AM-7PM
            Sunday 7AM-7PM

        2 - Morning Rush:
            Monday-Friday 7AM-9:30AM

        3 - Evening Rush:
            Monday-Friday 4PM-7PM

        4 - Weekday Day:
            Monday-Friday 9:30AM-4PM

        5 - Weekday Night:
            Sunday-Thursday 7PM-Midnight
            Monday-Friday Midnight-7AM
    """

    def weekday_class(hour,minute):
        if hour < 7 or hour >= 19:
            return 5
        elif hour >= 7 and (hour < 9 or hour == 9 and minute < 30):
            return 2
        elif hour >= 16 and hour <= 19:
            return 3
        else:
            return 4

    def friday_class(hour,minute):
        if hour < 19:
            return weekday_cat(hour_minute)
        else:
            return 0

    def saturday_class(hour,minute):
        if hour < 7 or hour >= 19:
            return 0
        else:
            return 1

    def sunday_class(hour,minute):
        if hour < 7:
            return 0
        elif hour < 19:
            return 1
        else: return 5

    trip_id2time_class = {}

    for trip_id in trip_id2time.keys():
        day,hour,minute = trip_id2time[trip_id]
        trip_class = 0
        if day < 4:
            trip_class = weekday_class(hour,minute)
        elif day == 4:
            trip_class = friday_class(hour,minute)
        elif day == 5:
            trip_class = saturday_class(hour,minute)
        else:
            trip_class = sunday_class(hour,minute)

        trip_id2time_class[trip_id] = trip_class

        print "(%d %d:%d): %d" % (day, hour, minute, time_class)


    with open('trip_id2class.pickle','wb') as output:
        pickle.dump(trip_id2time_class,output)

def main():
    to_time_fn = 'trip_id2time.pickle'
    trip_id2time = pickle.load(open(to_time_fn,'rb'))
    #create_mappings('cab_chronological.txt',to_time_filename)
    #analyze_times(trip_id2time)
    classify_trips(trip_id2time)




if __name__ == '__main__':
    main()
