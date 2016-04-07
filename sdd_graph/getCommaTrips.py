import sys
import buildGraphs as bg
import pickle

def writeTrip(trip_num,lines,out,trip_id2line_num):
    #found = 0
    i = trip_id2line_num[trip_num]
    while bg.normalize_simple(lines[i])[0] == trip_num:
        out.write(lines[i])
        i += 1
    """
    for line in orig:
        num = bg.normalize_simple(line)[0]
        #num = int(line[:5])
        if tripNum == num:
            found = 1
            out.write(line)
            #print line
        elif found == 1:
            break
    """
	
#orig = open('csvGPS.txt','r')
#orig = open('cab_trips.txt','r')
trip_id2line_num = pickle.load(open('trip_id2line_num.pickle','rb'))
orig = open('cab_chronological.txt','r')
lines = orig.readlines()
orig.close()
out = open('getTripsOut.txt','w')
trip_nums = map(int,sys.argv[1].split(','))
for trip_num in trip_nums: 
    writeTrip(trip_num,lines,out,trip_id2line_num)
out.close()

