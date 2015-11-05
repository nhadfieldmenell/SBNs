#trips[][4]: start lat, start long, end lat, end long
#There is no trip 15707--it skips from 15706 to 15708


def normalize(line):
	tripNum = int(line[:5])
	latitude = float(line[32:41])
	longitude = float(line[42:-1])
	return tripNum,latitude,longitude


def createTrips(fn):
	#1-indexed array
	#first entry (trips[0]) is all 0's
	#done to stay consistent with tripIds
	trips = [[0 for x in range(4)] for x in range(25001)]
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
		else:
			trips[index][2] = parsed[1]
			trips[index][3] = parsed[2]
		counter += 1
	return trips


def minMax(trips):
	maxLat = float(0)
	minLat = float(500)
	maxLong = float(-500)
	minLong = float(0)
	
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
			
		if trip[1] < minLong:
			minLong = trip[1]
		elif trip[1] > maxLong:
			maxLong = trip[1]
		if trip[3] < minLong:
			minLong = trip[3]
		elif trip[3] > maxLong:
			maxLong = trip[3]
	return maxLat, minLat, maxLong, minLong

	
#create arrays of buckets, where each bucket holds a lat/long 
#and all trips within latLongStep of that point
def createStartEnd(latLongStep):
	#each of these arrays hold an array whose first value is the midpoint
	#of a latitude or longitude line (with radius latLongStep)
	#the subsuquent values are tripIds that fall on that line
	startLat = []
	startLong = []
	endLat = []
	endLong = []
	lat = minLat
	while lat < maxLat:
		current = []
		current2 = []
		current.append(lat)
		startLat.append(current)
		endLat.append(current2)
		lat += latLongStep
		
	lon = minLong
	while lon < maxLong:
		current = []
		current2 = []
		current.append(lon)
		startLong.append(current)
		endLong.append(current2)
		lon += latLongStep
		
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
		adjIndex = int(adjStartLat/latLongStep)
		if startLat[adjIndex].count(index) == 0:
			startLat[adjIndex].append(index)
			addedTo = adjIndex	
		if adjIndex+1 < maxLat and startLat[adjIndex+1].count(index) == 0:
			startLat[adjIndex+1].append(index)
			
		adjEndLat = trip[2] - minLat
		adjIndex = int(adjEndLat/latLongStep)
		if endLat[adjIndex].count(index) == 0:
			endLat[adjIndex].append(index)
		if adjIndex+1 < maxLat and endLat[adjIndex+1].count(index) == 0:
			endLat[adjIndex+1].append(index)
			
		adjStartLong = trip[1] - minLong
		adjIndex = int(adjStartLong/latLongStep)
		if startLong[adjIndex].count(index) == 0:
			startLong[adjIndex].append(index)
		if adjIndex+1 < maxLong and startLong[adjIndex+1].count(index) == 0:
			startLong[adjIndex+1].append(index)
			
		adjEndLong = trip[3] - minLong
		adjIndex = int(adjEndLong/latLongStep)
		if endLong[adjIndex].count(index) == 0:
			endLong[adjIndex].append(index)
		if adjIndex+1 < maxLong and endLong[adjIndex+1].count(index) == 0:
			endLong[adjIndex+1].append(index)
		index += 1
		
	return startLat, startLong, endLat, endLong


def findGoodPoints(latArray,longArray,numPoints):
	bestStartLat = 0
	bestStartLong = 0
	numMatches = 0
	latIndex = 0
	fewestMatches = 0
	fewestIndex = 0
	#array of 2-tuples
	#first tuple is a 3-tuple: matches, startlat, startlong
	#second tuple is all of the tripId's that start in that position
	goodPoints = []
	for i in range(numPoints):
		point = []
		triple = [0,0,0]
		point.append(triple)
		blank = []
		point.append(blank)
		goodPoints.append(point)	
		
	#finds the lat-long square of radius latLongStep that has the most uber pickups
	for lats in latArray:
		longIndex = 0
		for longs in longArray:
			matches = 0
			matchArray = []
			for latI in lats:
				if longs.count(latI) == 1:
					matches += 1
					matchArray.append(latI)
					
			if matches > fewestMatches:
				point = []
				triple = [matches,latIndex,longIndex]
				point.append(triple)
				point.append(matchArray)
				goodPoints[fewestIndex] = point
				
				fewestMatches = goodPoints[0][0][0]
				fewestIndex = 0
				
				for i in range(numPoints):
					if goodPoints[i][0][0] < fewestMatches:
						fewestMatches = goodPoints[i][0][0]
						fewestIndex = i
						
			longIndex += 1
			
		latIndex += 1
	return goodPoints

orig = open('firstLast.txt','r')

#trips[][4]: start lat, start long, end lat, end long
trips = createTrips(orig)

#calc the min and max latitute and longitude traversed in system
minMaxRet = minMax(trips)
maxLat = minMaxRet[0]
minLat = minMaxRet[1]
maxLong = minMaxRet[2]
minLong = minMaxRet[3]


latLongStep = 0.003
startEndRet = createStartEnd(latLongStep)
startLat = startEndRet[0]
startLong = startEndRet[1]
endLat = startEndRet[2]
endLong = startEndRet[3]


numPoints = 7
goodStarts = findGoodPoints(startLat,startLong,numPoints)
goodEnds = findGoodPoints(endLat,endLong,numPoints)

	

for point in goodStarts:
	print point

#for track in goodStarts[6][1]:
#	print str(track)+','+str(trips[track][0])

			
"""
	
#for i in range(20):
#	print startLat[i]
	
#for i in range(20):
#	print endLat[i]
#	
#for i in range(20):
#	print startLong[i]
	
#for i in range(20):
#	print startLong[i]


"""
	
#print str(minLat) + ',' + str(maxLat) + ',' + str(minLong) + ',' + str(maxLong)