import tensorflow as tf
import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from imutils.video import FPS
from .centroidtracker import CentroidTracker
from .trackableobject import TrackableObject
import base64

class CountModel:
    def __init__(self):
        detect_fn = tf.saved_model.load("saved_model")
        self.detect_fn = detect_fn
    
    def predict(self, image_64_decode):
      
        PATH_VIDEO = "video_in.mp4"
        video_result = open(PATH_VIDEO, "wb")
        video_result.write(base64.b64decode(image_64_decode))
        
        PATH_OUTPUT = "output.mp4"

        SKIP_FPS = 30
        TRESHOLD = 0.5

        vs = cv2.VideoCapture(PATH_VIDEO)

        writer = None

        W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ct = CentroidTracker(maxDisappeared= 40, maxDistance = 50)

        trackers = []
        trackableObjects = {}

        totalFrame = 0
        totalDown = 0
        totalUp = 0

        DIRECTION_PEOPLE = True

        POINT = [0, int((H/2)-H*0.1), W, int(H*0.1)]

        fps = FPS().start()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(PATH_OUTPUT, fourcc, 20.0, (W, H), True)

        while True:
            ret, frame = vs.read()

            if frame is None:
                break

            status = "Waiting"
            rects = []


            if totalFrame % SKIP_FPS == 0:
                status = "Detecting"
                trackers = []

                image_np = np.array(frame)

                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]

                detections = self.detect_fn(input_tensor)

                detection_scores = np.array(detections["detection_scores"][0])

                detection_clean = [x for x in detection_scores if x >= TRESHOLD]

                for x in range(len(detection_clean)):
                    idx = int(detections['detection_classes'][0][x])

                    ymin, xmin, ymax, xmax = np.array(detections['detection_boxes'][0][x])
                    box = [xmin, ymin, xmax, ymax] * np.array([W,H, W, H])

                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(frame, rect)

                    trackers.append(tracker)
            else:
                for tracker in trackers:

                    status = "Tracking"

                    tracker.update(frame)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    rects.append((startX, startY, endX, endY))

            cv2.rectangle(frame, (POINT[0], POINT[1]), (POINT[0]+ POINT[2], POINT[1] + POINT[3]), (255, 0, 255), 2)

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)

                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    if not to.counted:
                        if centroid[0] > POINT[0] and centroid[0] < (POINT[0]+ POINT[2]) and centroid[1] > POINT[1] and centroid[1] < (POINT[1]+POINT[3]):
                            if DIRECTION_PEOPLE:
                                if direction >0:
                                    totalUp += 1
                                    to.counted = True
                                else:
                                    totalDown +=1
                                    to.counted = True
                            else:
                                if direction <0:
                                    totalUp += 1
                                    to.counted = True
                                else:
                                    totalDown +=1
                                    to.counted = True

                trackableObjects[objectID] = to

                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

            info = [
                  ("Up", totalUp),
                  ("Down", totalDown),
                  ("Status", status),
            ]

            for (i, (k,v)) in enumerate(info):
                text = "{}: {}".format(k,v)
                cv2.putText(frame, text, (10, H - ((i*20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            writer.write(frame)

            totalFrame += 1
            fps.update()

        fps.stop()

        print("Time completed {}".format(fps.elapsed()))
        print("Time per frame {}".format(fps.fps()))

        writer.release()

        vs.release()
        
        video = open(PATH_OUTPUT, "rb")
        video_read = video.read()
        image_64_encode = base64.b64encode(video_read)
        image_64_encode_return = image_64_encode.decode() 
        return image_64_encode_return