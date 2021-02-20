import numpy as np
import cv2
from cv2utils.camera import make_camera_with_args

def process1(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    # img = cv2.dilate(img, None, iterations=2)

    cnts, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, cnts, -1, (255, 0, 0), 3)
    orig = frame
    frame = np.zeros_like(frame)
    cv2.fillPoly(frame, cnts, (255, 255, 255))
    frame = np.where(frame < 255, orig, frame)
    return frame

def process2(frame):
    purple = np.zeros_like(frame)
    purple[:] = (255, 0, 255)
    s = (purple.shape[0], purple.shape[1], 1)
    i = np.tile(np.all(frame < 230, axis=2).reshape(s), (1, 1, 3))
    return np.where(i, frame, purple)

def process3(frame):
    return np.where(frame > 150, frame * 2, frame)
    # return frame * 2

def process4(frame):
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    output = np.zeros_like(frame)
    for i in range(len(frame) - 3):
        for j in range(len(frame[i]) - 3):
            output[i, j, 0] = np.sum(kernel * frame[i:i+3, j:j+3, 0])
            output[i, j, 1] = np.sum(kernel * frame[i:i+3, j:j+3, 1])
            output[i, j, 2] = np.sum(kernel * frame[i:i+3, j:j+3, 2])
    return output

def prepare(process=None):
    def f(frame):
        frame = cv2.flip(frame, 1)
        
        if process != None:
            frame = process(frame)

        return frame
    return f

camera, args = make_camera_with_args(cam=1)
camera.stream(
    prepare=prepare(process4),
    log=True
)