orig = open('origGPS.txt','r')
csv = open('csvGPS.txt','w')
for line in orig:
	for char in line:
		if char != '	':
			csv.write(char)
		else:
			csv.write(',')