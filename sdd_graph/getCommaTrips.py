import sys
import buildGraphs as bg

def writeTrip(tripNum,orig,out):
    found = 0
    for line in orig:
        num = bg.normalize_simple(line)[0]
        #num = int(line[:5])
        if tripNum == num:
            found = 1
            out.write(line)
            #print line
        elif found == 1:
            break
	
#orig = open('csvGPS.txt','r')
#orig = open('cab_trips.txt','r')
orig = open('cab_chronological.txt','r')
out = open('getTripsOut.txt','w')
numArgs = len(sys.argv)
for i in range(1,numArgs):
    writeTrip(int(sys.argv[i][:-1]),orig,out)

