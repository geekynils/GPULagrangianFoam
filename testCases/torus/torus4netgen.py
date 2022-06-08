#!/usr/bin/python

# Thanks to Josef Buergler for this.

# from numpy import *;

from math import cos, sin, pi

x0 = 0.0;
y0 = 2.0;
r  = 1.0;
n  = 50;

print "algebraic3d";
print ""; 
print "curve2d curve01 = (",n,";";
for i in range(0,n):
    print x0 + r*cos(-2*pi/n*i),",",y0 + r*sin(-2*pi/n*i),";"

print n,";";
for i in range(0,n-1):
    print "2,",i+1,",",i+2,";";
print "2,",n,",",1,");";
print "";
print "solid outer = revolution (0, 0, 0; 1, 0, 0; curve01) -maxh = 0.2;";
print "";
print "tlo outer;";
