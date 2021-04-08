import cv2

face_cascade = cv2.CascadeClassifier(
    "./classifiers/haarcascade_frontalface_default.xml"
)

def find_faces(num_of_people):
    def f(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = list(face_cascade.detectMultiScale(gray))
        sorted(faces, key=lambda face: -face[2] * face[3])
        return faces[:num_of_people]
    return f

def full(frame):
    h, w, d = frame.shape
    return [(0, 0, w, h)]