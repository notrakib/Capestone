from django.shortcuts import render
from .realtime import image_pred
from .models import *
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
# import winsound


def index(request):
    return render(request, "app.html")


@gzip.gzip_page
def start(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'app.html')


def stop(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(stop_camera(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'app.html')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        return image_pred(image)

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def stop_camera(camera):
    camera.__del__()


def gen(camera):
    face10 = []
    eye10 = []
    while True:
        frame, current_status, current_status_eye = camera.get_frame()
        face10.append(current_status)
        if len(face10) == 10:
            # for i in range(0, 6):
            #     if face10[i] == 'Yawn' and face10[i+1] == 'Yawn' and face10[i+2] == 'Yawn' and face10[i+3] == 'Yawn':
            #         winsound.Beep(1500, 500)
            #     elif face10[i] == 'Anger' and face10[i+1] == 'Anger' and face10[i+2] == 'Anger' and face10[i+3] == 'Anger':
            #         winsound.Beep(300, 500)
            #     elif face10[i] == 'Happy' and face10[i+1] == 'Happy' and face10[i+2] == 'Happy' and face10[i+3] == 'Happy':
            #         winsound.Beep(700, 500)

            face10 = []

        eye10.append(current_status_eye)
        if len(eye10) == 10:
            # for j in range(0, 6):
            #     if eye10[j] == 'Closed_Eyes' and eye10[j+1] == 'Closed_Eyes' and eye10[j+2] == 'Closed_Eyes' and eye10[j+3] == 'Closed_Eyes':
            #         winsound.Beep(2500, 500)
            #     if eye10[j] != 'Open_Eyes' and eye10[j+1] != 'Open_Eyes' and eye10[j+2] != 'Open_Eyes' and eye10[j+3] != 'Open_Eyes':
            #         winsound.Beep(2500, 500)

            eye10 = []

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
