import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)    
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
pipeline.start(config)

COCO_KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def main():
    # Load the YOLOv8 Pose model
    model = YOLO("yolo11n-pose.pt")  # You can choose other models like yolov8s-pose.pt, etc.

    # Open the webcam
    # cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

    # if not cap.isOpened():
    #     print("Error: Cannot access the camera.")
    #     return

    # print("Press 'q' to quit.")
    
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # ret, frame = cap.read()
        # if not ret:
        #     print("Error: Cannot read from the camera.")
        #     break
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print(depth_image.shape)
        # deph_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
        # depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_GRAY2RGB)
        # Detect poses in the frame
        results = model(color_image)
        # print(results[0].keypoints.xy)
        
        persons = []
        for idx, result in enumerate(results[0].keypoints):
            print(f"Person {idx + 1} Keypoints:")
            
            # Initialize an empty list to store the person's keypoints
            person_keypoints = []
            keys = result.xy.tolist()[0]
            # Extract x, y coordinates and confidence for each keypoint
            for kp_idx in range(len(keys)):  
                x,y = keys[kp_idx]
                # print(depth_image((x,y)))             
                person_keypoints.append((int(x), int(y)))

            # Print the keypoints for this person
            for x, y  in person_keypoints:
                print(f" (x={x}, y={y})")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Show the frame
        cv2.imshow("Pose Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

