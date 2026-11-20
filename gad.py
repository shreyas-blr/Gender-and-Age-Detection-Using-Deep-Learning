import cv2
import argparse
import os
import sys

# Resolve absolute path to project directory for models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    # Ensure line thickness is at least 1 (avoid OpenCV error on small images)
    lineThickness = max(1, int(round(frameHeight / 150)))

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * frameWidth))
            y1 = max(0, int(detections[0, 0, i, 4] * frameHeight))
            x2 = min(frameWidth - 1, int(detections[0, 0, i, 5] * frameWidth))
            y2 = min(frameHeight - 1, int(detections[0, 0, i, 6] * frameHeight))
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), lineThickness, 8)
    return frameOpencvDnn, faceBoxes


def process_frame(frame, faceNet, ageNet, genderNet, ageList, genderList, MODEL_MEAN_VALUES, padding=20):
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):
                     min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Prevent empty face crop crashes
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        text_y = max(24, faceBox[1] - 10)
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return resultImg


def main():
    parser = argparse.ArgumentParser(description="Gender and Age Detection using OpenCV DNN")
    parser.add_argument('--image', help="Path to input image file (omit for webcam)")
    parser.add_argument('--conf', type=float, default=0.7, help="Confidence threshold for face detection")
    args = parser.parse_args()

    # Paths to model weights and architectures
    faceProto = os.path.join(SCRIPT_DIR, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(SCRIPT_DIR, "opencv_face_detector_uint8.pb")
    ageProto = os.path.join(SCRIPT_DIR, "age_deploy.prototxt")
    ageModel = os.path.join(SCRIPT_DIR, "age_net.caffemodel")
    genderProto = os.path.join(SCRIPT_DIR, "gender_deploy.prototxt")
    genderModel = os.path.join(SCRIPT_DIR, "gender_net.caffemodel")

    # Verify model files exist
    for f in [faceProto, faceModel, ageProto, ageModel, genderProto, genderModel]:
        if not os.path.exists(f):
            print(f"Error: Model file '{f}' not found!")
            sys.exit(1)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load DNN networks
    try:
        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
    except Exception as e:
        print(f"Error loading models with OpenCV DNN: {e}")
        sys.exit(1)

    # --- Mode 1: Static Image ---
    if args.image:
        image_path = os.path.abspath(args.image) if not os.path.isabs(args.image) else args.image
        if not os.path.exists(image_path):
            print(f"Error: Image file '{args.image}' not found.")
            sys.exit(1)

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Failed to decode image '{args.image}'.")
            sys.exit(1)

        resultImg = process_frame(frame, faceNet, ageNet, genderNet, ageList, genderList, MODEL_MEAN_VALUES)
        cv2.imshow("Detecting age and gender", resultImg)
        print("Press any key on the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # --- Mode 2: Webcam Stream ---
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam (index 0).")
        print("Tip: If on macOS, ensure Camera permissions are granted to your Terminal / IDE in System Settings.")
        print("You can also run on a sample image instead: python gad.py --image girl1.jpg")
        sys.exit(1)

    print("Camera active. Press 'q' or 'Esc' on the video window to quit.")
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            print("Warning: Could not read frame from camera.")
            break

        resultImg = process_frame(frame, faceNet, ageNet, genderNet, ageList, genderList, MODEL_MEAN_VALUES)
        cv2.imshow("Detecting age and gender", resultImg)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
