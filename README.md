# Road-Accidents-Reduction

## Problem Statement
Conceptualise and design a solution to reduce the accidents on highways

## Proposed Solution

# Road-Accidents-Reduction

Our ideas to the given problem (Problem 9):

**Sleepiness Detection (Driver Distraction/Unawareness)** - We can place a camera just infront of the driver to detect sleepiness and if sleepiness is detected, the system can generate an alert to wake up the driver whilst decelerating the vehicle.

**Vehicle to Vehicle Communication (Unexpected behaviour of nearby vehicles)** - We can establish a communication channel among the vehicle in their respective range/vicinity via GPS so as to get information such as speed, accelaration and distance of the other vehicle from the current one. Thus helping the current vehicle to accordingly manage its speed and distance from the surrounding vehicles.

**Vehicle Preboot diagnosis (Avoid Vehicle Internal Failures)** - During ignition, we can perform a system diagnosis on various parts of the car for example, the air pressure in tyres, alignment of tyres, possible engine failures, leakages, etc.

**Obstacle Detection (Collision avoidance in case of less visibility)** - Using the proximity and distance sensor to find any object in the vehicle's vicinity in less visibility conditions to the driver for example, during snowfall, rainy days, foggy days etc. This will again alert the driver and slow down the vehicle.


## Procedure to run Sleepiness_Detection

Necessary Dependencies
* numpy
* dlib
* imutils
* cv2
* scipy

Execute `python Sleepiness_Detection.py` to run the script
