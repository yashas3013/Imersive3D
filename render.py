import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import json
from flask import Flask, jsonify
import threading
from flask_cors import CORS
from scipy.spatial.transform import Rotation as R

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Store keypoints data in memory
keypoints_data = []

COCO_KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Depth camera parameters
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_min = 0.11  # meters
depth_max = 1.0   # meters

depth_itr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_itr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(
    profile.get_stream(rs.stream.color))
color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(
    profile.get_stream(rs.stream.depth))

# Flask route to send keypoints
@app.route('/keypoints', methods=['GET'])
def get_keypoints():
    return jsonify(keypoints_data)

def compute_orientation(joints):
    """Calculate the orientation of the body using shoulders and hips."""
    try:
        left_shoulder = np.array(joints["Left Shoulder"])
        right_shoulder = np.array(joints["Right Shoulder"])
        left_hip = np.array(joints["Left Hip"])
        right_hip = np.array(joints["Right Hip"])

        # Forward direction (torso)
        forward_vec = (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
        forward_vec /= np.linalg.norm(forward_vec)  # Normalize

        # Right direction (shoulders)
        right_vec = right_shoulder - left_shoulder
        right_vec /= np.linalg.norm(right_vec)

        # Upward direction (cross product)
        up_vec = np.cross(right_vec, forward_vec)
        up_vec /= np.linalg.norm(up_vec)

        # Construct rotation matrix
        rotation_matrix = np.vstack([right_vec, up_vec, forward_vec]).T

        # Convert to quaternion
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # (x, y, z, w)

        return quaternion.tolist()  # Return as list (x, y, z, w)
    
    except KeyError:
        return [0, 0, 0, 1]  # Default no rotation (identity quaternion)

def save_keypoints(person_keypoints):
    """Save keypoints and compute orientation."""
    global keypoints_data
    joints = {COCO_KEYPOINT_NAMES[i]: keypoint for i, keypoint in enumerate(person_keypoints)}

    # Compute orientation from shoulders and hips
    orientation = compute_orientation(joints)

    keypoints_data = [
        {
            "name": COCO_KEYPOINT_NAMES[i],
            "x": keypoint[0],  # Right
            "y": keypoint[2],  # Down
            "z": -keypoint[1]    # Forward
        }
        for i, keypoint in enumerate(person_keypoints)
    ]

    # Add orientation (quaternion) to data
    keypoints_data.append({
        "name": "Orientation",
        "x": orientation[0],
        "y": orientation[1],
        "z": orientation[2],
        "w": orientation[3]
    })

    print("Keypoints and orientation updated.")

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# Start Flask server in a separate thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

def main():
    # Load the YOLOv8 Pose model
    model = YOLO("yolo11n-pose.pt")  # You can choose another model like yolov8s-pose.pt
    
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Error, No data from camera")
            break
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth image to 8-bit for visualization
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
        
        # Detect poses in the frame
        results = model(color_image)
        
        persons = []
        for idx, result in enumerate(results[0].keypoints):
            print(f"Person {idx + 1} Keypoints:")
            
            person_keypoints = []
            keys = result.xy.tolist()[0]
            
            for kp_idx in range(len(keys)):  
                x, y = keys[kp_idx]
                if result.conf[0][kp_idx].item() < 0.7:
                    continue

                # Convert color pixel to depth pixel
                depth_point = rs.rs2_project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(), depth_scale, depth_min, depth_max,
                    depth_itr, color_itr, depth_to_color_extrin, color_to_depth_extrin, [x, y]
                )

                # Get depth value and convert to 3D coordinates
                depth = depth_image[int(depth_point[1]), int(depth_point[0])]
                point3d = rs.rs2_deproject_pixel_to_point(depth_itr, depth_point, depth * depth_scale)
                
                print(COCO_KEYPOINT_NAMES[kp_idx], point3d)

                cv2.circle(depth_colormap, (int(depth_point[0]), int(depth_point[1])), 5, (255, 0, 0), -1)
                person_keypoints.append(point3d)
            
            if person_keypoints:
                save_keypoints(person_keypoints)
        
        # Visualize the results
        annotated_frame = results[0].plot()
        cv2.imshow("Pose Detection", annotated_frame)
        cv2.imshow("Depth", depth_colormap)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
