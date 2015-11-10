import math
import numpy

#trips[][7]: start lat, start Lon, end lat, end Lon, start[day,hour,minute,second], end[day,hour,minute,second], dist (in miles)
#There is no trip 0 or 15707--it skips from 15706 to 15708

#parse trip id, latitude, and longitude from a raw line of data
def normalize(line):
	tripNum = int(line[:5])
	latitude = float(line[32:41])
	Lonitude = float(line[42:-1])
	time = convertTimestamp(line[6:25])
	return tripNum,latitude,Lonitude,time


#pass in a timestamp in form 2007-01-06T08:42:58
#return a quadruple: day of week (0:sunday,1:monday,2:tuesday,...,6:saturday),hour,minute,second
def convertTimestamp(stamp):
	day = int(stamp[9])
	if day == 7:
		day = 0
	hour = int(stamp[11:13])
	minute = int(stamp[14:16])
	second = int(stamp[17:19])
	return day,hour,minute,second


#read in trips from file that contains trip start on one line, trip end on next line
def createTrips(fn):
	#1-indexed array
	#first entry (trips[0]) is all 0's
	#done to stay consistent with tripIds
	trips = [[0 for x in range(7)] for x in range(25001)]
	counter = 0
	index = 0
	for line in orig:
		parsed = normalize(line)
		if counter%2 == 0:
			index += 1
			
			#only happens to skip 15707
			if parsed[0] != index:
				index += 1
			trips[index][0] = parsed[1]
			trips[index][1] = parsed[2]
			trips[index][4] = parsed[3]
		else:
			trips[index][2] = parsed[1]
			trips[index][3] = parsed[2]
			trips[index][5] = parsed[3]
			trips[index][6] = gpsDif(trips[index][0],trips[index][2],trips[index][1],trips[index][3])
		counter += 1
	return trips


#create a list that contains full info about trip
#full[][4]: full[][0] is trip day, full[][1][0] is trip start hour, full[][1][1] is start minute, 
#full[][2] is trip duration(minutes), full[][3] is a list of all gps coordinates (full[][3][0] is latitude, full[][3][1] is longitude)
def createFull(fn):
	

#calc the min and max latitute and Lonitude traversed in system
def minMax(trips):
	maxLat = float(0)
	minLat = float(500)
	maxLon = float(-500)
	minLon = float(0)
	
	for trip in trips:
		if trip[0] == 0 or trip[1] == 0 or trip[2] == 0 or trip[3] == 0:
			continue
			
		if trip[0] < minLat:
			minLat = trip[0]
		elif trip[0] > maxLat:
			maxLat = trip[0]
		if trip[2] < minLat:
			minLat = trip[2]
		elif trip[2] > maxLat:
			maxLat = trip[2]
			
		if trip[1] < minLon:
			minLon = trip[1]
		elif trip[1] > maxLon:
			maxLon = trip[1]
		if trip[3] < minLon:
			minLon = trip[3]
		elif trip[3] > maxLon:
			maxLon = trip[3]
	return maxLat, minLat, maxLon, minLon

	
#create arrays of buckets, where each bucket holds a lat/Lon 
#and all trips within latStep/lonStep of that point
def createStartEnd(latStep,lonStep):
	#each of these arrays hold an array whose first value is the midpoint
	#of a latitude or Lonitude line (with radius latLonStep)
	#the subsuquent values are tripIds that fall on that line
	startLat = []
	startLon = []
	endLat = []
	endLon = []
	lat = minLat
	while lat <= maxLat:
		current = []
		current2 = []
		current.append(lat)
		startLat.append(current)
		endLat.append(current2)
		lat += latStep
		
	lon = minLon
	while lon <= maxLon:
		current = []
		current2 = []
		current.append(lon)
		startLon.append(current)
		endLon.append(current2)
		lon += lonStep
		
	index = 0
	addedTo = 0
	for trip in trips:
		if index == 0:
			index += 1
			continue
			
		if trip[0] == 0 or trip[1] == 0 or trip[2] == 0 or trip[3] == 0:
			index += 1
			continue
			
		adjStartLat = trip[0] - minLat
		adjIndex = int(adjStartLat/latStep)
		if startLat[adjIndex].count(index) == 0:
			startLat[adjIndex].append(index)
			addedTo = adjIndex
			
		adjEndLat = trip[2] - minLat
		adjIndex = int(adjEndLat/latStep)
		if endLat[adjIndex].count(index) == 0:
			endLat[adjIndex].append(index)
			
		adjStartLon = trip[1] - minLon
		adjIndex = int(adjStartLon/lonStep)
		if startLon[adjIndex].count(index) == 0:
			startLon[adjIndex].append(index)
			
		adjEndLon = trip[3] - minLon
		adjIndex = int(adjEndLon/lonStep)
		if endLon[adjIndex].count(index) == 0:
			endLon[adjIndex].append(index)
	
		index += 1
		
	return startLat, startLon, endLat, endLon


#finds points that have the most overlapping trips (either starts or ends)
def findGoodPoints(latArray,LonArray,numPoints):
	bestStartLat = 0
	bestStartLon = 0
	numMatches = 0
	latIndex = 0
	fewestMatches = 0
	fewestIndex = 0
	#array of 2-tuples
	#first tuple is a 3-tuple: matches, startlat, startLon
	#second tuple is all of the tripId's that start in that position
	goodPoints = []
	for i in range(numPoints):
		point = []
		triple = [0,0,0]
		point.append(triple)
		blank = []
		point.append(blank)
		goodPoints.append(point)	
		
	#finds the lat-Lon square of radius latLonStep that has the most uber pickups
	for lats in latArray:
		lonIndex = 0
		for lons in LonArray:
			matches = 0
			matchArray = []
			for latI in lats:
				if lons.count(latI) == 1:
					matches += 1
					matchArray.append(latI)
					
			if matches > fewestMatches:
				point = []
				triple = [matches,latIndex,lonIndex]
				point.append(triple)
				point.append(matchArray)
				goodPoints[fewestIndex] = point
				
				fewestMatches = goodPoints[0][0][0]
				fewestIndex = 0
				
				for i in range(numPoints):
					if goodPoints[i][0][0] < fewestMatches:
						fewestMatches = goodPoints[i][0][0]
						fewestIndex = i
						
			lonIndex += 1
			
		latIndex += 1
	return goodPoints


#always rounds down
#converts gps to bin number in lat/lon arrays
def convertGPS(lat,lon,latStep,lonStep,minLat,minLon):
	latInt = int((lat - minLat)/latStep)
	lonInt = int((lon - minLon)/lonStep)
	return latInt,lonInt


#pass in bucket numbers for GPS points rather than actual points
def findTrips(startLatNum,startLonNum,endLatNum,endLonNum,startLats,startLons,endLats,endLons,trips):
	validTrips = []
	for tripId in startLats[startLatNum]:
		if float(tripId).is_integer():
			if endLons[endLonNum].count(tripId) == 1:
				if endLats[endLatNum].count(tripId) == 1:
					if startLons[startLonNum].count(tripId) == 1:
						validTrips.append(tripId)
	return validTrips
	
	


#pass in a list of id's that all have the same start region
#find the end region(s) that hold the most end points from those trips
#startLatPos is index in the startLat array of the starting latitude
#minDist will allow you to input a minimum distance (in miles) for start-end point pairs
def findBestEnd(trips,startIds,startLatPos,startLonPos,latStep,lonStep,minLat,minLon,numLats,numLons,minDist,gridSize):
	diffTrips = []
	for trip in trips:
		tripInfo = []
		startInfo = convertGPS(trip[0],trip[1],latStep,lonStep,minLat,minLon)
		endInfo = convertGPS(trip[2],trip[3],latStep,lonStep,minLat,minLon)
		tripInfo.append(startInfo[0])
		tripInfo.append(startInfo[1])
		tripInfo.append(endInfo[0])
		tripInfo.append(endInfo[1])
		diffTrips.append(tripInfo)
		
	#dests is a 2D array representing the grid of the city each grid space holds the 
	#indices of the trips that started at the start point and end at that grid space
	dests = [[0 for x in range(numLons)] for x in range(numLats)]
	destArray = [[[] for x in range(numLons)] for x in range(numLats)]
	for tripNum in startIds:
		theTrip = diffTrips[tripNum]
		dests[theTrip[2]][theTrip[3]] += 1
		destArray[theTrip[2]][theTrip[3]].append(tripNum)
		
	#print dests
	best = 0
	index = [0,0]
	secondBest = 0
	secondIndex = [0,0]
	minDistBest = 0
	minDistIndex = [0,0]
	
	#alrightIndicies holds all end points that have at 
	#least alrightMatchNum trips originating at the start point
	alrightIndicies = []
	alrightMatchNum = 10;
	for i in range(numLats):
		for j in range(numLons):
			if dests[i][j] > alrightMatchNum:
				alrightIndicies.append(index)
			if dests[i][j] > secondBest:
				if dests[i][j] > best:
					secondBest = best
					secondIndex[0] = index[0]
					secondIndex[1] = index[1]
					best = dests[i][j]
					index = [i,j]
				else:
					secondBest = dests[i][j]
					secondIndex = [i,j]
			if dests[i][j] > minDistBest:
				if math.sqrt(math.pow(i-startLatPos,2)+math.pow(j-startLonPos,2)) >= minDist:
					minDistBest = dests[i][j]
					minDistIndex[0] = i
					minDistIndex[1] = j
					
		
	print "best"
	print best
	print index
	print destArray[index[0]][index[1]]
	print "\nsecond best"	
	print secondBest
	print secondIndex
	print destArray[secondIndex[0]][secondIndex[1]]
	print "\nminDist best"
	print minDistBest
	print minDistIndex
	print destArray[minDistIndex[0]][minDistIndex[1]]
	return alrightIndicies
		
	


#finds the number of ride starts per time of day
#select granularity with hourInc (how many hours each block is)
#returns list of 3 tuples: start time, end time, # of rides
#24 should be divisible by hourInc
def ridesByHour(trips,hourInc):
	numPeriods = 24/hourInc
	tripsPerTime = [[0 for x in range(3)] for x in range(numPeriods)]
	for trip in trips:
		# because 1-indexed
		if trip[0] == 0:
			continue
		tripsPerTime[trip[4][1]/hourInc][2]+=1
	for i in range(numPeriods):
		startHour = i*hourInc
		endHour = (i+1)*hourInc
		tripsPerTime[i][0] = startHour
		tripsPerTime[i][1] = endHour
	
	best = 0
	bestPeriod = [0,0]
	tot = 0			
	for i in range(numPeriods):
		tot += tripsPerTime[i][2]
		if tripsPerTime[i][2] > best:
			best = tripsPerTime[i][2]
			bestPeriod = [tripsPerTime[i][0],tripsPerTime[i][1]]
	
	print "tot"
	print tot
	print "best period"		
	print best
	print bestPeriod
	for i in range(numPeriods):
		print str(i) + ": " + str(tripsPerTime[i][2])
	return tripsPerTime


#finds the # of rides per time period per day
#returns list of list of 3 tuples: tripsPerPeriod[i][j][k]:
#	i selects day, j selects hour range, k selects from 3 tuple: start time, end time, # of rides
def ridesByHourAndDay(trips,hourInc):
	numPeriods = 24/hourInc
	tripsPerPeriod = [[[0 for x in range(3)] for x in range(numPeriods)] for x in range(7)]
	for trip in trips:
		# because 1-indexed
		if trip[0] == 0:
			continue
		tripsPerPeriod[trip[4][0]][trip[4][1]/hourInc][2]+=1
	for day in range(7):	
		for i in range(numPeriods):
			startHour = i*hourInc
			endHour = (i+1)*hourInc
			tripsPerPeriod[day][i][0] = startHour
			tripsPerPeriod[day][i][1] = endHour
	
	tot = 0		
	for day in range(7):
		for i in range(numPeriods):
			#print str(day) + "," + str(i) + ": " + str(tripsPerPeriod[day][i][2])
			tot += tripsPerPeriod[day][i][2]
		
	numTops = 10
	#4-tuples: day, period start hour, period end hour, # of trips
	#holds these tuples for the numTops-th best periods
	topPeriods = [[0 for x in range(4)] for x in range(numTops)]
	minTrips = 0
	minIndex = 0
	for day in range(7):
		for period in range(numPeriods):
			if tripsPerPeriod[day][period][2] > minTrips:
				topPeriods[minIndex] = [day,tripsPerPeriod[day][period][0],tripsPerPeriod[day][period][1],tripsPerPeriod[day][period][2]]
				
				#find new lowest
				minTrips = tripsPerPeriod[day][period][2]
				for topIndex in range(numTops):
					if topPeriods[topIndex][3] <= minTrips:
						minTrips = topPeriods[topIndex][3]
						minIndex = topIndex
	print "tot: " + str(tot)
	print "top periods"
	for i in topPeriods:
		print i


#pass in coordinates for 2 points
#return distance (in miles) between the two
def gpsDif(lat1,lat2,lon1,lon2):
	latDif = lat2-lat1
	lonDif = lon2-lon1
	latDist = latDif/0.0145
	lonDist = lonDif/0.01825
	return math.sqrt(math.pow(latDist,2)+math.pow(lonDist,2))	

#pass in trips
#return an array holding the length of each trip
#array indices do not match up to trip numbers because of no trip 0 or trip 15707
def tripLengths(trips):
	tripLengths = []
	for i in range(len(trips)):
		if trips[i] == 0:
			continue
		tripLengths.append(trips[i][6])
		
	print "average distance (miles): " + str(numpy.mean(tripLengths))
	print "25th percentile: " + str(numpy.percentile(tripLengths,25))	
	print "50th percentile: " + str(numpy.percentile(tripLengths,50))
	print "75th percentile: " + str(numpy.percentile(tripLengths,75))
	print "90th percentile: " + str(numpy.percentile(tripLengths,90))
	return tripLengths


orig = open('firstLast.txt','r')
fullFn = open('csvGps.txt','r')

#trips[][7]: start lat, start Lon, end lat, end Lon, start[day,hour,minute,second], end[day,hour,minute,second], dist (in miles)
trips = createTrips(orig)

tripLengths(trips)

#print trips[2000]

#tripLengths(trips)

"""
ridesByHour(trips,1)
ridesByHourAndDay(trips,2)

minMaxRet = minMax(trips)
maxLat = minMaxRet[0]
minLat = minMaxRet[1]
maxLon = minMaxRet[2]
minLon = minMaxRet[3]

print minLat
print maxLat
print minLon
print maxLon

#enter the square edge length to specify grid regions (in miles)
#1 mile is approximately 0.0145 in latitude in SF area
#1 mile is approximately 0.01825 in longitude in SF area
gridSize = 0.5
latStep = 0.0145*gridSize
lonStep = 0.01825*gridSize


startEndRet = createStartEnd(latStep,lonStep)
startLat = startEndRet[0]
startLon = startEndRet[1]
endLat = startEndRet[2]
endLon = startEndRet[3]

specStartLat = float(37.788929)
specStartLon = float(-122.399598)
specEndLat = float(37.754739)
specEndLon = float(  -122.415317)
startPoint = convertGPS(specStartLat,specStartLon,latStep,lonStep,minLat,minLon)
endPoint = convertGPS(specEndLat,specEndLon,latStep,lonStep,minLat,minLon)

#theTrips = findTrips(startPoint[0],startPoint[1],endPoint[0],endPoint[1],startLat,startLon,endLat,endLon,trips)
#print theTrips
"""
"""
convGPS = convertGPS(37.774, -122.4152,latLonStep,latLonStep,minLat,minLon)
print convGPS[0]
print convGPS[1]

print startLat[80][0]
print startLon[41][0]
"""
""""
numPoints = 1
goodStarts = findGoodPoints(startLat,startLon,numPoints)
#goodEnds = findGoodPoints(endLat,endLon,numPoints)

bestEnds = findBestEnd(trips,goodStarts[0][1],goodStarts[0][0][1],goodStarts[0][0][2],latStep,lonStep,minLat,minLon,len(startLat),len(startLon),4,gridSize)



print "starts"
for point in goodStarts:
	print point[0]
	print startLat[point[0][1]][0]
	print startLon[point[0][2]][0]
	
"""
"""	
print "ends"
for point in goodEnds:
	print point[0]
	

#for track in goodStarts[6][1]:
#	print str(track)+','+str(trips[track][0])
		

	
#for i in range(20):
#	print startLat[i]
	
#for i in range(20):
#	print endLat[i]
#	
#for i in range(20):
#	print startLon[i]
	
#for i in range(20):
#	print startLon[i]


"""
	
#print str(minLat) + ',' + str(maxLat) + ',' + str(minLon) + ',' + str(maxLon)