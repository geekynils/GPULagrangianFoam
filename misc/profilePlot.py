from sys import argv
import csv
from pyx import *
from subprocess import Popen, PIPE, STDOUT

def parseFile(path):
    
    r = csv.reader(open(path, 'rb'))

    # No way to specify comments?!
    r.next()
    r.next()
    r.next()
    r.next()
    r.next()

    # fieldnames = r.next()
    fieldnames = [ "gpustarttimestamp", "method", "gputime", "cputime" ]

    lineDict = {}
    data = []

    for line in r:
        i = 0
        # Ensure that a new object is created here
        lineDict = {}
        for name in fieldnames:
            lineDict[name] = line[i]
            i += 1
        data.append(lineDict)
        
    return data


def normalizeTimestamp(data):
    """gpustarttimestamp is in hex, unsinged integer. Units are 
    nanoseconds. gputime and cputime However are stored in 
    microseconds therfore the division by 1000"""

    beginning = int(data[0]["gpustarttimestamp"], 16)

    for line in data:
        line["gpustarttimestamp"] = \
            (int(line["gpustarttimestamp"], 16) - beginning) / 1000


def demangleFuncNames(data):
    """Demangles C++ names and removes arguments and brackets"""
    
    filtred = []
    
    for line in data:
        wordInput = line["method"]
        if wordInput.startswith("memcpy"):
            output = wordInput
        else:
            p = Popen(['c++filt', '-t', '-p'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
            output = p.communicate(input=wordInput)[0]

        # Workaround for thrust stuff
        if output.startswith("thrust"):
            output = "thrust"
        
        line["method"] = output
        filtred.append(output)
            
    return filtred


def mapColors(namesList):

    colorList = [ "#FFFF00", "#DF0174", "#FE2EC8", "#5882FA", "#08088A", \
                  "#FE2E2E", "#B40404", "#40FF00", "#BDBDBD", "#A9F5BC" ]
    names = set(namesList)
    colorDict = {}
    i = 0
    for name in names:
        colorDict[name] = colorList[i]
        i += 1
    return colorDict
    

def widthPlot(data, colorDict, outfile):

    c = canvas.canvas()
    
    # Set up LaTex
    text.set(mode="latex")
    text.preamble(r"\usepackage{color}")
    
    # Key: hex, Val: stringForTex
    latexColorDict = {}
    colorStringLatex = ""
    
    i = 0
    for colorHex in colorDict.values():
        colorStringLatex = "a" + str(i)
        col = color.rgbfromhexstring(colorHex)
        text.preamble(r"\definecolor{%s}{rgb}{%s, %s, %s}" % \
            (colorStringLatex, col.color['r'], col.color['g'], col.color['b']))
        latexColorDict[colorHex] = colorStringLatex
        i += 1
        
    yPos = 0
    
    methodNamesList = []
    
    #TODO This can be done in one line, just don't know how..
    for line in data:
        methodNamesList.append(line["method"])
        
    methodNames = set(methodNamesList)
    
    for methodName in methodNames:
        colorStringLatex = latexColorDict[colorDict[methodName]]
        c.text(0, yPos, r"\textbf{\textcolor{%s}{%s}}" % (colorStringLatex, methodName))
        yPos += 0.5
        
    plotWidth = 20
    plotHeight = 2

    totExecTime = getTotExecTime(data)
    
    for line in data:
    
        begin = float(line["gpustarttimestamp"]) / totExecTime * plotWidth
        end =  float(line["gputime"]) / totExecTime * plotWidth
        
        col = color.rgbfromhexstring(colorDict[line["method"]])
        
        c.fill(path.rect(begin, yPos, end, plotHeight), [col])
    
    c.writeEPSfile(outfile)

def getTotExecTime(data):

    totTime = 0
    record = data[len(data)-1]
    totTime = record["gpustarttimestamp"] + float(record["gputime"])
    return totTime

data = parseFile("/home/nils/CUDAProf/calcLambdasTorus_Session7_Context_0.csv")
normalizeTimestamp(data)
namesList = demangleFuncNames(data)
colorDict = mapColors(namesList)
getTotExecTime(data)
widthPlot(data, colorDict, "widthPlot")
