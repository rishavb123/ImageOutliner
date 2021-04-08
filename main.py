from cv2utils.args import make_parser
import numpy as np
import cv2
from cv2utils.camera import make_camera_with_args
from kernels import *
from choose_sections import *

def process(frame):
    sections = choose_section(frame)

    processed = cv2.filter2D(frame, -1, kernel)

    for x, y, w, h in sections:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        frame[y : y + h, x : x + w] = processed[y : y + h, x : x + w]


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
    parser=parser, cam=0, log=True, res=(640, 360), fps=30
)
num_of_people = args.people

kernel = OUTLINER
choose_section = full

camera.stream(prepare=prepare(process))
