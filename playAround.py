#trips[][4]: start lat, start Lon, end lat, end Lon
#There is no trip 15707--it skips from 15706 to 15708


def normalize(line):
	tripNum = int(line[:5])
	latitude = float(line[32:41])
	Lonitude = float(line[42:-1])
	return tripNum,latitude,Lonitude


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
#and all trips within latLonStep of that point
def createStartEnd(latLonStep):
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
		lat += latLonStep
		
	lon = minLon
	while lon <= maxLon:
		current = []
		current2 = []
		current.append(lon)
		startLon.append(current)
		endLon.append(current2)
		lon += latLonStep
		
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
		adjIndex = int(adjStartLat/latLonStep)
		if startLat[adjIndex].count(index) == 0:
			startLat[adjIndex].append(index)
			addedTo = adjIndex
		"""
		if adjIndex+1 < maxLat:
			if startLat[adjIndex+1].count(index) == 0:
				startLat[adjIndex+1].append(index)
		"""
			
		adjEndLat = trip[2] - minLat
		adjIndex = int(adjEndLat/latLonStep)
		if endLat[adjIndex].count(index) == 0:
			endLat[adjIndex].append(index)
		"""
		if adjIndex+1 < maxLat:
			if endLat[adjIndex+1].count(index) == 0:
				endLat[adjIndex+1].append(index)
		"""
			
		adjStartLon = trip[1] - minLon
		adjIndex = int(adjStartLon/latLonStep)
		if startLon[adjIndex].count(index) == 0:
			startLon[adjIndex].append(index)
		"""
		if adjIndex+1 < maxLon:
			if startLon[adjIndex+1].count(index) == 0:
				startLon[adjIndex+1].append(index)
		"""
			
		adjEndLon = trip[3] - minLon
		adjIndex = int(adjEndLon/latLonStep)
		if endLon[adjIndex].count(index) == 0:
			endLon[adjIndex].append(index)
		"""
		if adjIndex+1 < maxLon:
			if endLon[adjIndex+1].count(index) == 0:
				endLon[adjIndex+1].append(index)
		"""
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
def findBestEnd(trips,startIds,latStep,lonStep,minLat,minLon,numLats,numLons):
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
		
	dests = [[0 for x in range(numLons)] for x in range(numLats)]
	destArray = [[[] for x in range(numLons)] for x in range(numLats)]
	for tripNum in startIds:
		theTrip = diffTrips[tripNum]
		dests[theTrip[2]][theTrip[3]] += 1
		destArray[theTrip[2]][theTrip[3]].append(tripNum)
		
	print dests
	best = 0
	index = [0,0]
	secondBest = 0
	secondIndex = [0,0]
	alrightIndicies = []
	for i in range(numLats):
		for j in range(numLons):
			if dests[i][j] > 10:
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
					
		
	print destArray[index[0]][index[1]]		
	print best
	print index
	print "\n"
	print destArray[secondIndex[0]][secondIndex[1]]	
	print secondBest
	print secondIndex
	return alrightIndicies
		
	


orig = open('firstLast.txt','r')

#trips[][4]: start lat, start Lon, end lat, end Lon
trips = createTrips(orig)

minMaxRet = minMax(trips)
maxLat = minMaxRet[0]
minLat = minMaxRet[1]
maxLon = minMaxRet[2]
minLon = minMaxRet[3]


latLonStep = 0.005
startEndRet = createStartEnd(latLonStep)
startLat = startEndRet[0]
startLon = startEndRet[1]
endLat = startEndRet[2]
endLon = startEndRet[3]

specStartLat = float(37.788929)
specStartLon = float(-122.399598)
specEndLat = float(37.754739)
specEndLon = float(  -122.415317)
startPoint = convertGPS(specStartLat,specStartLon,latLonStep,latLonStep,minLat,minLon)
endPoint = convertGPS(specEndLat,specEndLon,latLonStep,latLonStep,minLat,minLon)

#theTrips = findTrips(startPoint[0],startPoint[1],endPoint[0],endPoint[1],startLat,startLon,endLat,endLon,trips)
#print theTrips

"""
convGPS = convertGPS(37.774, -122.4152,latLonStep,latLonStep,minLat,minLon)
print convGPS[0]
print convGPS[1]

print startLat[80][0]
print startLon[41][0]
"""

numPoints = 1
goodStarts = findGoodPoints(startLat,startLon,numPoints)
#goodEnds = findGoodPoints(endLat,endLon,numPoints)

bestEnds = findBestEnd(trips,goodStarts[0][1],latLonStep,latLonStep,minLat,minLon,len(startLat),len(startLon))



print "starts"
for point in goodStarts:
	print point[0]
	print startLat[point[0][1]][0]
	print startLon[point[0][2]][0]
"""		
print "ends"
for point in goodEnds:
	print point[0]
	

#for track in goodStarts[6][1]:
#	print str(track)+','+str(trips[track][0])

"""			
"""
	
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