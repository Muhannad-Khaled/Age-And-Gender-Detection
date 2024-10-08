import cv2
import numpy as np

# Function to detect faces
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold for face detection
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)

            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return frame, bboxs

# Load the pre-trained models for face, age, and gender detection
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Initialize the networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define mean values and lists for age and gender
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(21, 26)', '(28, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
genderList = ['Male', 'Female']

# Initialize video capture
video = cv2.VideoCapture(0)
padding = 20

# Queue for smoothing age predictions
age_queue = []

while True:
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)

    for bbox in bboxs:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender prediction
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        genderConfidence = max(genderPred[0])
        if genderConfidence > 0.8:  # Gender confidence threshold
            gender = genderList[genderPred[0].argmax()]

        # Age prediction with smoothing
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        predicted_age = ageList[agePred[0].argmax()]

        age_queue.append(predicted_age)
        if len(age_queue) > 5:  # Smoothing window of 5 frames
            age_queue.pop(0)

        # Get the most frequent age prediction in the window
        smoothed_age = max(set(age_queue), key=age_queue.count)

        label = f"{smoothed_age}, {gender}"

        # Draw label and bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Age-Gender Detection", frame)

    # Break the loop if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

