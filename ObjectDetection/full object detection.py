import cv2

# Load the pre-trained SSD model
net = cv2.dnn.readNetFromTensorflow("C:/Users/esska/Downloads/frozen_inference_graph.pb","C:/Users/esska/Downloads/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

# Read class names from the coco.names file
with open("C:/Users/esska/Downloads/coco.names", "r") as f:
    classNames = f.read().strip().split("\n")

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform object detection
    detections = net.forward()

    # Display the detected objects on the frame
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Minimum confidence threshold for detection
            class_id = int(detections[0, 0, i, 1])

            if class_id >= 0 and class_id < len(classNames):
                class_name = classNames[class_id]
            else:
                class_name = "Unknown"

            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x, y, w, h) = box.astype("int")

            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()



