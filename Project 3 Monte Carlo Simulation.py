#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:01:41 2018

@author: aiyuan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


#================================
# Uniform Random Number Generator
#================================
s = np.random.uniform(size=60000) # size divisible by 2 and 3

# show uniform frequency of s
plt.hist(s,bins=50,color='skyblue',ec='black')
plt.ylabel('$Frequency$')
plt.xlabel('$x$')
plt.show()

# group values in s into groups of twos to get (x,y) coordinates
x = s[0::2]
y = s[1::2]
plt.scatter(x,y,s=0.2)
plt.ylabel('y')
plt.xlabel('x')
plt.show()

# group values in s into groups of threes to get (x,y,z) coordinates
x = s[0::3]
y = s[1::3]
z = s[2::3]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,s=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#=================================================
# Function of exponential random number generator
#=================================================
def randomExpo(n,lumda):
    # generates n numbers distributed according to exponential function
    # e^(-x/lumda)
    # returns array of n numbers
    i = np.random.uniform(size=n)
    s = -lumda*np.log(i)
    return s

# code to test the function
s = randomExpo(1000,45)
(n, bins, patches) = plt.hist(s,bins=50,color='skyblue',ec='black')
plt.ylabel('Frequency')
plt.xlabel('x / cm')
plt.show()

# code to test the function more rigorously
# by verifying that characteristic attenuation length in water is about 45cm
# if there is no scattering (only absorption) and mean free path input is 45cm
    
# array to store calculated attenuation length values
attenLengthValues = np.array([]) 

# run 100 times to get a mean value for attenution length and its estimated error
for x in range(100):
    s = randomExpo(10000,45)
    (n, bins, patches) = plt.hist(s,bins=50,color='skyblue',ec='black')
    plt.show()

    # store values of wanted counts and middle x-value of wanted bins
    n_new = np.array([])
    x_new = np.array([])

    # discard bins with zero counts and take middle x-value of bins 
    for i in range(len(n)):
        if n[i] != 0:
            n_new = np.append(n_new, n[i])
            x_value = (bins[i] + bins[i+1]) / 2
            x_new = np.append(x_new, x_value) 

    # plot x_new against ln(n_new)
    (m, c) = np.polyfit(np.log(n_new), x_new, 1)
    plt.scatter(np.log(n_new), x_new, marker='+')
    plt.ylabel('$x / cm$')
    plt.xlabel('$ln N(x)$')
    plt.show()
    
    attenLengthValues = np.append(attenLengthValues, -m)
    
# calculate mean, standard deviation, and standard error of mean for attenLengthValues
std = np.std(attenLengthValues)
mean = np.mean(attenLengthValues)
mean_error = std/math.sqrt(100)

#=====================================================
# Function that generates uniformly distributed points
# on the surface of unit sphere
#=====================================================
def randomSpherePoints(n):
    # generates n points on a unit sphere of radius 1
    # returns tuple of (x_array,y_array,z_array) coordinates of each point
    r = 1     
    phi = np.random.uniform(0,2*math.pi,size=n)
    costheta = np.random.uniform(-1,1,size=n)
    theta = np.arccos( costheta )
    
    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )
    
    return (x,y,z)

# code to test the function
(x,y,z) = randomSpherePoints(10000)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z,s=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#=======================================================
# Function that generates 3D isotropic steps distributed 
# exponentially with a mean free path lumda
#======================================== ==============
def randomIsoSteps(n,lumda):
    # generates n 3D isotropic steps with exponentially distributed length 
    # and a mean free path of lumda
    # returns tuple of (x_array,y_array,z_array) for the steps
    r = randomExpo(n,lumda)
    phi = np.random.uniform(0,2*math.pi,size=n)
    costheta = np.random.uniform(-1,1,size=n)
    theta = np.arccos( costheta )
    
    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )

    return (x,y,z)    

# code to test the function
(x,y,z) = randomIsoSteps(10000,45)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z,s=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#========================================================================
# Function that simulates firing neutrons at a medium of finite thickness 
#========================================================================
# medium of finite thickness with existing between x=0 and x=L and
# infinite length in y and z

def fireNeutrons(n,L,mfp,aProb,sProb):
    # simulate firing 'n' neutrons at a medium of thickness 'L' in cm with mean free path 'mfp' in cm,
    # abosrption probabilty 'aProb' in decimal and scattering probability 'sProb' in decimal
    # returns (no. of neutrons transmitted, no. of neutrons reflected, no. of neutrons absorbed)
    
    aCount = 0    # parameter to count number of neutrons absorbed
    rCount = 0    # parameter to count number of neutrons reflected
    tCount = 0    # parameter to count number of neutrons transmitted
            
    while n > 0:
        # create arrays to store position of neutron and give initial position of (0,0,0)
        (xPos,yPos,zPos) = (np.array([0]),np.array([0]),np.array([0]))
        
        # create temporary variables to store temporary x,y,z component values of each step
        (xVect,yVect,zVect) = (0,0,0)
            
        aCheck = 0     # parameter to check whether absorption has happened
        rCheck = 0     # parameter to check whether reflection has happened
        tCheck = 0     # parameter to check if transmitted
        i = 0          # parameter to count number of steps 
        dice = 0       # parameter to determine absorbed or scattered after every step
        result = ''    # parameter to announce result of the neutron in the end
        
        while aCheck == 0 and rCheck == 0 and tCheck == 0 :
            if i == 0:
                # first step always travels in x direction
                (xVect,yVect,zVect) = (randomExpo(1,mfp),0,0)
                (xPos, yPos, zPos) = (np.append(xPos,xVect),np.append(yPos,yVect),np.append(zPos,zVect))
                i += 1
            else:
                (xVect,yVect,zVect) = randomIsoSteps(1,mfp)
                (xPos,yPos,zPos) = (np.append(xPos,xPos[i]+xVect),np.append(yPos,yPos[i]+yVect),np.append(zPos,zPos[i]+zVect))
                i += 1
                
            if xPos[i] < 0:
                # check if reflected
                rCheck = 1
            
            if xPos[i] > L:
                # check if transmitted
                tCheck = 1
            
            if rCheck == 0 and tCheck == 0:
                # if not reflected and not transmitted
                dice = np.random.uniform()   # 'roll a dice'
                if dice <= aProb:
                    # check if aborbed
                    aCheck = 1
        
        # record final result of this neutron             
        if rCheck == 1: 
            rCount += 1 
            result = 'Reflected'
        if tCheck == 1: 
            tCount += 1 
            result = 'Transmitted'
        if aCheck == 1: 
            aCount += 1 
            result = 'Absorbed'
        
        n -= 1   # done with 1 neutron simultaion
    
        # plot an example 3D random walk history of the last neutron fired
        if n == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.plot(xPos[:1],yPos[:1],zPos[:1],'o-',c='green',label="Start",zorder=2)
            ax.plot(xPos[-1:],yPos[-1:],zPos[-1:],'o-',c='coral',label=result,zorder=2)
            ax.plot(xPos,yPos,zPos,'o-',markersize=5,zorder=1)
            ax.set_xlabel('$x / cm$')
            ax.set_ylabel('$y / cm$')
            ax.set_zlabel('$z / cm$')
            ax.legend()
            plt.show()
        
    # optional code to print results        
    print('No. of neutrons fired',tCount+rCount+aCount)
    print('No. transmitted', tCount)
    print('No. reflected', rCount)
    print('No. absorbed', aCount)
    
    return (tCount,rCount,aCount)

#======================================================================================
# Function that calculates percentage of transmission/reflection/absorption with errors
#======================================================================================
def runSimulations(n,runs,L,mfp,aProb,sProb):
    # runs fireNeutrons() simulaion 'runs' times to get percentage of transmission/reflection/absorption
    # with corresponding percentage errors
    # returns and print the calculated values
    tResults = np.array([])    # array to store transmission number of all runs
    rResults = np.array([])    # array to store reflection number of all runs
    aResults = np.array([])    # array to store absorption number of all runs
    
    runsCount = 0     # parameter to count no. of runs elapsed
    
    while runsCount < runs:
        # fire the neutrons
        (tCount,rCount,aCount) = fireNeutrons(n,L,mfp,aProb,sProb)
        # record the results
        (tResults,rResults,aResults) = (np.append(tResults,tCount),np.append(rResults,rCount),np.append(aResults,aCount))
        # done with 1 run
        runsCount += 1
    
    # calculate mean 
    (tMean,rMean,aMean) = (np.mean(tResults),np.mean(rResults),np.mean(aResults))
    # calculate standard deviation 
    (tStd,rStd,aStd) = (np.std(tResults),np.std(rResults),np.std(aResults))
    # calculate standard error of mean
    (tMeanErr,rMeanErr,aMeanErr) = (tStd/math.sqrt(runs-1),rStd/math.sqrt(runs-1),aStd/math.sqrt(runs-1))
    # calculate percentage of transmission/reflection/absorption
    (tPercent,rPercent,aPercent) = (tMean/n*100,rMean/n*100,aMean/n*100)
    # calculate percentage errors of transmission/reflection/absorption
    (tPercentErr,rPercentErr,aPercentErr) = (tMeanErr/n*100,rMeanErr/n*100,aMeanErr/n*100)
    
    print('Transmission %.2f +- %.2f %.2f +- %.2f' % (tMean,tMeanErr,tPercent,tPercentErr),'%')
    print('Reflection %.2f +- %.2f %.2f +- %.2f' % (rMean,rMeanErr,rPercent,rPercentErr),'%')
    print('Absorption %.2f +- %.2f %.2f +- %.2f' % (aMean,aMeanErr,aPercent,aPercentErr),'%')
    return ( [tMean,tMeanErr,tPercent,tPercentErr],[rMean,rMeanErr,rPercent,rPercentErr], [aMean,aMeanErr,aPercent,aPercentErr])
    
#==================================================
# Simulations For Medium of Water (Thickness 10cm )
#==================================================
# vary no. of neutrons simulated to see the variation of errors
    
# array to store the n values to use
nValues = np.array([100,500,1000,2000,3000,4000,5000,6000,7000,8000])
# arrays to store transmission/reflection/absorption data collected
(tDataWater,rDataWater,aDataWater) = ([],[],[])

# simulate for all the nValues
for i in nValues:
    data = runSimulations(n=i,runs=10,L=10,mfp=0.29,aProb=0.006,sProb=0.994)
    tDataWater.append(data[0])
    rDataWater.append(data[1])
    aDataWater.append(data[2])

# convert into numpy arrays
(tDataWater,rDataWater,aDataWater) = (np.array(tDataWater),np.array(rDataWater),np.array(aDataWater))

plt.errorbar(nValues,tDataWater[:,3],label='Transmission',linestyle='dotted',marker='s',markersize=5)
plt.errorbar(nValues,rDataWater[:,3],label='Reflection',linestyle='dotted',marker='s',markersize=5)
plt.errorbar(nValues,aDataWater[:,3],label='Absorption',linestyle='dotted',marker='s',markersize=5)
plt.xlabel('No. of Neutrons Simulated')
plt.ylabel('Percentage Error (%)')
plt.legend()

#==========================================================
# Simulations for Medium of Medium of Lead (Thickness 10cm)
#==========================================================
# vary no. of neutrons simulated to see the variation of errors
    
# array to store the n values to use
nValues = np.array([100,500,1000,2000,3000,4000,5000,6000,7000,8000])
# arrays to store transmission/reflection/absorption data collected
(tDataLead,rDataLead,aDataLead) = ([],[],[])

# simulate for all the nValues
for i in nValues:
    data = runSimulations(n=i,runs=10,L=10,mfp=2.67,aProb=0.014,sProb=0.986)
    tDataLead.append(data[0])
    rDataLead.append(data[1])
    aDataLead.append(data[2])

# convert into numpy arrays
(tDataLead,rDataLead,aDataLead) = (np.array(tDataLead),np.array(rDataLead),np.array(aDataLead))

plt.errorbar(nValues,tDataLead[:,3],label='Transmission',linestyle='dotted',marker='s',markersize=5)
plt.errorbar(nValues,rDataLead[:,3],label='Reflection',linestyle='dotted',marker='s',markersize=5)
plt.errorbar(nValues,aDataLead[:,3],label='Absorption',linestyle='dotted',marker='s',markersize=5)
plt.xlabel('No. of Neutrons Simulated')
plt.ylabel('Percentage Error (%)')
plt.legend()
plt.show()

#==============================================================
# Simulations for Medium of Medium of Graphite (Thickness 10cm)
#==============================================================
# vary no. of neutrons simulated to see the variation of errors
    
# array to store the n values to use
nValues = np.array([100,500,1000,2000,3000,4000,5000,6000,7000,8000])
# arrays to store transmission/reflection/absorption data collected
(tDataGraphite,rDataGraphite,aDataGraphite) = ([],[],[])

# simulate for all the nValues
for i in nValues:
    data = runSimulations(n=i,runs=10,L=10,mfp=2.52,aProb=0.001,sProb=0.999)
    tDataGraphite.append(data[0])
    rDataGraphite.append(data[1])
    aDataGraphite.append(data[2])

# convert into numpy arrays
(tDataGraphite,rDataGraphite,aDataGraphite) = (np.array(tDataGraphite),np.array(rDataGraphite),np.array(aDataGraphite))

plt.errorbar(nValues,tDataGraphite[:,3],label='Transmission',linestyle='dotted',marker='s',markersize=5)
plt.errorbar(nValues,rDataGraphite[:,3],label='Reflection',linestyle='dotted',marker='s',markersize=5)
plt.errorbar(nValues,aDataGraphite[:,3],label='Absorption',linestyle='dotted',marker='s',markersize=5)
plt.xlabel('No. of Neutrons Simulated')
plt.ylabel('Percentage Error (%)')
plt.legend()
plt.show()

#=====================================================
# Simulations for Water of Varying Thickness (1-10cm)    
#=====================================================
# fire 6000 neutrons, run 10 times for each thickness to get t,r,a values

# array to store the thickness values of water to use (1-10cm)
thickWater = np.array([1,2,3,4,5,6,7,8,9,10])

# arrays to store transmission/reflection/absorption values calculated
(tValuesWater,rValuesWater,aValuesWater) = ([],[],[])

# simulate for all the thickWater values
for i in thickWater:
    data = runSimulations(n=6000,runs=10,L=i,mfp=0.29,aProb=0.006,sProb=0.994)
    tValuesWater.append(data[0])
    rValuesWater.append(data[1])
    aValuesWater.append(data[2])

# convert into numpy arrays
(tValuesWater,rValuesWater,aValuesWater) = (np.array(tValuesWater),np.array(rValuesWater),np.array(aValuesWater))

plt.errorbar(thickWater,tValuesWater[:,2],yerr=tValuesWater[:,3],label='Transmission',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.errorbar(thickWater,rValuesWater[:,2],yerr=rValuesWater[:,3],label='Reflection',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.errorbar(thickWater,aValuesWater[:,2],yerr=aValuesWater[:,3],label='Absorption',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.legend()
plt.xlabel('Thickness (cm)')
plt.ylabel('Percentage (%)')


#=====================================================
# Simulations for Lead of Varying Thickness (10-100cm)    
#=====================================================
# fire 4000 neutrons, run 10 times for each thickness to get t,r,a values

# array to store the thickness values of lead to use (10-100cm)
thickLead = np.array([10,20,30,40,50,60,70,80,90,100])

# arrays to store transmission/reflection/absorption values calculated
(tValuesLead,rValuesLead,aValuesLead) = ([],[],[])

# simulate for all the thickLead values
for i in thickLead:
    data = runSimulations(n=4000,runs=10,L=i,mfp=2.67,aProb=0.014,sProb=0.986)
    tValuesLead.append(data[0])
    rValuesLead.append(data[1])
    aValuesLead.append(data[2])

# convert into numpy arrays
(tValuesLead,rValuesLead,aValuesLead) = (np.array(tValuesLead),np.array(rValuesLead),np.array(aValuesLead))

plt.errorbar(thickLead,tValuesLead[:,2],yerr=tValuesLead[:,3],label='Transmission',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.errorbar(thickLead,rValuesLead[:,2],yerr=rValuesLead[:,3],label='Reflection',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.errorbar(thickLead,aValuesLead[:,2],yerr=aValuesLead[:,3],label='Absorption',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.legend()
plt.xlabel('Thickness (cm)')
plt.ylabel('Percentage (%)')

#=====================================================
# Simulations for Graphite of Varying Thickness (10-150)    
#=====================================================
# fire 5000 neutrons, run 10 times for each thickness to get t,r,a values

# array to store the thickness values of graphite to use (10-150cm)
thickGraphite = np.array([10,30,50,70,90,110,130,150])

# arrays to store transmission/reflection/absorption values calculated
(tValuesGraphite,rValuesGraphite,aValuesGraphite) = ([],[],[])

# simulate for all the thickGraphite values
for i in thickGraphite:
    data = runSimulations(n=5000,runs=10,L=i,mfp=2.52,aProb=0.001,sProb=0.999)
    tValuesGraphite.append(data[0])
    rValuesGraphite.append(data[1])
    aValuesGraphite.append(data[2])

# convert into numpy arrays
(tValuesGraphite,rValuesGraphite,aValuesGraphite) = (np.array(tValuesGraphite),np.array(rValuesGraphite),np.array(aValuesGraphite))

plt.errorbar(thickGraphite,tValuesGraphite[:,2],yerr=tValuesGraphite[:,3],label='Transmission',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.errorbar(thickGraphite,rValuesGraphite[:,2],yerr=rValuesGraphite[:,3],label='Reflection',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.errorbar(thickGraphite,aValuesGraphite[:,2],yerr=aValuesGraphite[:,3],label='Absorption',capsize=5,linestyle='dashed',marker='s',markersize=3)
plt.legend()
plt.xlabel('Thickness (cm)')
plt.ylabel('Percentage (%)')

#=================================================
# Find Characteristic Attenuation Length for Water
#=================================================
# plot ln(transmission percentage) against thickness
# gradient is -1/lumda where lumda is attenuation length
x = thickWater
y = np.log(tValuesWater[:,2])
yErr = tValuesWater[:,3] / tValuesWater[:,2]

# fit linear relationship
(p, V) = np.polyfit(x,y,1,cov=True)
(m, c) = (p[0],p[1])        # m is gradient, c is intercept
(mErr, cErr) = (np.sqrt(V[0][0]),np.sqrt(V[1][1])) 

# calculate chi-sq
chiSq = np.sum((np.polyval(p,x)-y)**2)
chiSqRed = chiSq / (len(x)-2)

# plot and print results
plt.plot(x,y,marker='v',linestyle='none')
plt.plot(x,np.polyval(p,x),linewidth=1)
plt.errorbar(x,y,yerr=yErr,fmt='none',capsize=5)
plt.ylabel('ln (Transmission %)')
plt.xlabel('Distance (cm)')
plt.show()

print('Chi Sq Reduced',chiSqRed)
print('Gradient',m,'+-',mErr)

#=================================================
# Find Characteristic Attenuation Length for Lead
#=================================================
# plot ln(transmission percentage) against thickness
# gradient is -1/lumda where lumda is attenuation length
x = thickLead
y = np.log(tValuesLead[:,2])
yErr = tValuesLead[:,3] / tValuesLead[:,2]

# fit linear relationship
(p, V) = np.polyfit(x,y,1,cov=True)
(m, c) = (p[0],p[1])        # m is gradient, c is intercept
(mErr, cErr) = (np.sqrt(V[0][0]),np.sqrt(V[1][1])) 

# calculate chi-sq
chiSq = np.sum((np.polyval(p,x)-y)**2)
chiSqRed = chiSq / (len(x)-2)

# plot and print results
plt.plot(x,y,marker='v',linestyle='none')
plt.plot(x,np.polyval(p,x),linewidth=1)
plt.errorbar(x,y,yerr=yErr,fmt='none',capsize=5)
plt.ylabel('ln (Transmission %)')
plt.xlabel('Distance (cm)')
plt.show()

print('Chi Sq Reduced',chiSqRed)
print('Gradient',m,'+-',mErr)

#====================================================
# Find Characteristic Attenuation Length for Graphite
#====================================================
# plot ln(transmission percentage) against thickness
# gradient is -1/lumda where lumda is attenuation length
x = thickGraphite
y = np.log(tValuesGraphite[:,2])
yErr = tValuesGraphite[:,3] / tValuesGraphite[:,2]

# fit linear relationship
(p, V) = np.polyfit(x,y,1,cov=True)
(m, c) = (p[0],p[1])        # m is gradient, c is intercept
(mErr, cErr) = (np.sqrt(V[0][0]),np.sqrt(V[1][1])) 

# calculate chi-sq
chiSq = np.sum((np.polyval(p,x)-y)**2)
chiSqRed = chiSq / (len(x)-2)

# plot and print results
plt.plot(x,y,marker='v',linestyle='none')
plt.plot(x,np.polyval(p,x),linewidth=1)
plt.errorbar(x,y,yerr=yErr,fmt='none',capsize=5)
plt.ylabel('ln (Transmission %)')
plt.xlabel('Distance (cm)')
plt.show()

print('Chi Sq Reduced',chiSqRed)
print('Gradient',m,'+-',mErr)











