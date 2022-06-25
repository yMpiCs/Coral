import time
import board
import digitalio
import pwmio
from adafruit_motor import servo
from shapely.geometry import box


import sys, tty, termios

pwm1 = pwmio.PWMOut(board.PWM3, duty_cycle=2 ** 15, frequency=50)
tiltServo = servo.Servo(pwm1)
tiltAngle=90

pwm2 = pwmio.PWMOut(board.PWM2, duty_cycle=2 ** 15, frequency=50)
panServo = servo.Servo(pwm2)
panAngle=90


def servo_movement(centroid):
    global panAngle
    global tiltAngle
    x, y = centroid
    #print("X,Y: ",x,y)
    cam_x, cam_y = 320, 240
    # xrange = 135-165, yrange = 220-260 
    if (x < 280):
        tiltAngle += 1
        if tiltAngle >  180:
            tiltAngle = 180
        tiltServo.angle = tiltAngle
    if (x > 360):
        tiltAngle -= 1
        if tiltAngle < 0:
            tiltAngle = 0
        tiltServo.angle = tiltAngle
    if (y < 190):
        panAngle -= 1
        if panAngle < 0:
            panAngle = 0
        panServo.angle = panAngle
    if (y > 290):
        panAngle += 1
        if panAngle > 180:
            panAngle = 180
        panServo.angle = panAngle

    # char = getch()
    
