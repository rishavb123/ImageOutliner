from cv2utils.args import make_parser
import numpy as np
import cv2
from cv2utils.camera import make_camera_with_args
from kernels import *

kernel = BLUR(7)
kernel = OUTLINER

face_cascade = cv2.CascadeClassifier(
    "./classifiers/haarcascade_frontalface_default.xml"
)


def process(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = list(face_cascade.detectMultiScale(gray))

    sorted(faces, key=lambda face: -face[2] * face[3])

    processed = cv2.filter2D(frame, -1, kernel)

    for x, y, w, h in faces[:num_of_people]:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        frame[y : y + h, x : x + w] = processed[y : y + h, x : x + w]

    frame = processed

    frame = cv2.resize(frame, dsize=(1920, 1080))

    return frame


def prepare(process=None):
    def f(frame):
        frame = cv2.flip(frame, 1)

        if process != None:
            frame = process(frame)

        return frame

    return f


parser = make_parser()
parser.add_argument(
    "-p",
    "--people",
    type=int,
    default=1,
    help="The number of people in the video stream",
)
camera, args = make_camera_with_args(
    parser=parser, cam=1, log=True, res=(640, 360), fps=30
)
num_of_people = args.people

camera.stream(prepare=prepare(process))
