# Road-Accidents-Reduction

## Problem Statement
Conceptualise and design a solution to reduce the accidents on highways

## Proposed Solution
Our ideas to the given problem (Problem 9):

**Sleepiness Detection (Driver Distraction/Unawareness)** - We can place a camera just infront of the driver to detect sleepiness and if sleepiness is detected, the system can generate an alert to wake up the driver whilst decelerating the vehicle.

**Vehicle to Vehicle Communication (Unexpected behaviour of nearby vehicles)** - We can establish a communication channel among the vehicle in their respective range/vicinity via GPS so as to get information such as speed, accelaration and distance of the other vehicle from the current one. Thus helping the current vehicle to accordingly manage its speed and distance from the surrounding vehicles.

**Vehicle Preboot diagnosis (Avoid Vehicle Internal Failures)** - During ignition, we can perform a system diagnosis on various parts of the car for example, the air pressure in tyres, alignment of tyres, possible engine failures, leakages, etc.

**Obstacle Detection (Collision avoidance in case of less visibility)** - Using the proximity and distance sensor to find any object in the vehicle's vicinity in less visibility conditions to the driver for example, during snowfall, rainy days, foggy days etc. This will again alert the driver and slow down the vehicle.


## Procedure to run `Sleepiness_Detection.py`
This code is written in Python 3. These dependencies will be needed to run the code:
* Numpy
* dlib
* imutils
* OpenCV
* Scipy

Execute `python Sleepiness_Detection.py` to run the script. 

### References
* The `.dat` file is taken from [here.](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
* EAR concept is taken from the work of [Soukupová and Čech](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf) in their 2016 paper.
