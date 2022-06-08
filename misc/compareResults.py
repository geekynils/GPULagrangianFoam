#! /usr/bin/python

import re

def drange(start, stop, step):
    "Range or xrange does not accept decimal steps."
    r = start
    while r < stop:
        yield r
        r += step

class Vec:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return "[ %f %f %f]" % (self.x, self.y, self.z)

class Particle:
    def __init__(self, pos=Vec(0,0,0), u=Vec(0,0,0), label=-1, time=0):
        self.pos = pos
        self.u = u
        self.label = label
        self.time = time
        
def captureLineByRe(string, regex):
    "Extracts the data on a line according to the given regex if it's not a comment"
    comment = "//"
    extractedData = []
    found = []
    for line in string.splitlines():
        if line.startswith(comment):
            continue   # TODO very cheap
        found = regex.findall(line)
        if len(found) != 0:
            v = Vec(found[0][0], found[0][1], found[0][2])   # TODO particle id is ignored so far
            extractedData.append(v)
    return extractedData

# Note the difference 
# positions: (x y z) particleID
# U: (x y z)

def parseVectors(string):
    "Extracts the data on a line according to the given regex if it's not a comment"
    comment = "//"
    extractedData = []
    found = []
    n = []
    particlePosRe = re.compile(r".*\(([-0-9e .]*)\).*")
    for line in string.splitlines():
        if line.startswith(comment):
            continue   # TODO very cheap
        found = particlePosRe.findall(line)
        if len(found) == 1:
            n = found[0].split()
            v = Vec(float(n[0]), float(n[1]), float(n[2]))
            extractedData.append(v)
    return extractedData
            
def printVecComparison(list1, list2):
    assert(len(list1)==len(list2))
    i=0
    while i<len(list1):
        print "[ %f %f %f ]" % (list1[i].x, list1[i].y, list1[i].z),
        print "[ %f %f %f ]" % (list2[i].x, list2[i].y, list2[i].z)
        i+=1

startTime = 0
endTime = 1
step = 0.01

cpuPos = [] 
gpuPos = []

for i in drange(startTime, endTime, step):
    f = open("resultsCPU/" + str(i) + "/lagrangian/defaultCloud/positions", 'r')
    cpuPos.append(parseVectors(f.read()))
    f.close()
    f = open("resultsGPU/" + str(i) + "/lagrangian/defaultCloud/positions", 'r')
    gpuPos.append(parseVectors(f.read()))
    f.close()

cpuVel = [] 
gpuVel = []
   
for i in drange(startTime, endTime, step):
    f = open("resultsCPU/" + str(i) + "/lagrangian/defaultCloud/U", 'r')
    cpuVel.append(parseVectors(f.read()))
    f.close()
    f = open("resultsGPU/" + str(i) + "/lagrangian/defaultCloud/U", 'r')
    gpuVel.append(parseVectors(f.read()))
    f.close()
    
# because we just have one particle per time step
i=0
while i < len(cpuPos):
    cpuPos[i] = cpuPos[i][0]
    gpuPos[i] = gpuPos[i][0]
    
    cpuVel[i] = cpuVel[i][0]
    gpuVel[i] = gpuVel[i][0]
    i+=1

print "Position data"
print "CPU                            GPU"
printVecComparison(cpuPos, gpuPos)
print

print "Velocity data"
print "CPU                            GPU"
printVecComparison(cpuVel, gpuVel)

