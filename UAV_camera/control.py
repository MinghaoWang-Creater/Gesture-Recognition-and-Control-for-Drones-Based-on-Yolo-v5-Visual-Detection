'''控制规则'''
# V ：           停止   0
# D 一根手指：    飞机向上1
# A 拳头：       飞机向下2
# W 三根手指：    飞机向前3
# 7 食指中指小指： 飞机向后4
# L 拇指食指：    飞机向右5
# Love：         飞机向左6
import cv2
# 摄像头设置
Camera_Width = 720
Camera_Height = 480
DetectRange = [6000, 11000]  # DetectRange[0] 是保持静止的检测人脸面积阈值下限，DetectRange[0] 是保持静止的检测人脸面积阈值上限
PID_Parameter = [0.5, 0.0004, 0.4]
pErrorRotate, pErrorUp = 0, 0

def trans(s):
    states = 0
    if s == '0: 448x640 1 up, ':
        states = 1
    elif s == '0: 448x640 1 down, ':
        states = 2
    elif s == '0: 448x640 1 forward, ':
        states = 3
    elif s == '0: 448x640 1 backward, ':
        states = 4
    elif s == '0: 448x640 1 right, ':
        states = 5
    elif s == '0: 448x640 1 left, ':
        states = 6
    return states


def control_Tello(s, speed):
    lr, fb, ud, yv = 0, 0, 0, 0
    ss = trans(s)
    if ss == 6:
        lr = -speed

    elif ss == 5:
        lr = speed

    if ss == 3:
        fb = speed
    elif ss == 4:
        fb = -speed

    if ss == 1:
        ud = speed
    elif ss == 2:
        ud = -speed

    #if kp.getKey("a"):
        #yv = -speed
    #elif kp.getKey("d"):
       # yv = speed

    return lr, fb, ud, yv