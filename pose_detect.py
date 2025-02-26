import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
from mpl_toolkits.mplot3d import Axes3D
import threading
import json

# initialize realsense pipeline
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)    
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
profile = pipeline.start(config)

COCO_KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Depth Scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

depth_min = 0.11 # meter
depth_max = 1.0  # meter

depth_itr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_itr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color))
color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth))



def save_keypoints_to_json(person_keypoints, filename="server/keypoints.json"):
    keypoints_data = []

    for i, keypoint in enumerate(person_keypoints):
        keypoints_data.append({
            "name": COCO_KEYPOINT_NAMES[i],
            "x": keypoint[0],  # Right
            "y": -keypoint[1], # Down
            "z": keypoint[2]   # Forward
        })

    # Save to a JSON file
    with open(filename, "w") as f:
        json.dump(keypoints_data, f, indent=4)

    print(f"Keypoints saved to {filename}")


def main():
    # Load the YOLOv8 Pose model
    model = YOLO("yolo11n-pose.pt")  # You can choose other models like yolov8s-pose.pt, etc.
    
    while True:
        frames = pipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame() 
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Error, No data from camera")
            break
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # convert from 8 bit to 16 bit
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
        
        # Detect poses in the frame
        results = model(color_image)
        
        persons = []
        for idx, result in enumerate(results[0].keypoints):
            print(f"Person {idx + 1} Keypoints:")
            
            # Initialize an empty list to store the person's keypoints
            person_keypoints = []
            keys = result.xy.tolist()[0]
            
            # Extract x, y coordinates and confidence for each keypoint
            for kp_idx in range(len(keys)):  
                x,y = keys[kp_idx]
                if (result.conf[0][kp_idx].item()<0.5):
                    continue
            
                # Convert color pixel to depth pixel
                depth_point = rs.rs2_project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(),depth_scale,depth_min,depth_max,depth_itr,color_itr,depth_to_color_extrin,color_to_depth_extrin,[x,y]
                )

                depth = depth_image[int(depth_point[1]),int(depth_point[0])] # numpy array
                
                point3d = rs.rs2_deproject_pixel_to_point(depth_itr,depth_point,depth*depth_scale)
                print(COCO_KEYPOINT_NAMES[kp_idx],point3d)

                cv2.circle(depth_colormap,(int(depth_point[0]),int(depth_point[1])),5,(255,0,0),-1)             
                person_keypoints.append(point3d)
            
            if person_keypoints:
                save_keypoints_to_json(person_keypoints)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Show the frame
        cv2.imshow("Pose Detection", annotated_frame)
        cv2.imshow("Depth",depth_colormap)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
