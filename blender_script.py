import bpy
import requests
import mathutils

URL = "http://localhost:5000/keypoints"  # Change this if needed
keypoint_objects = {}  # Store keypoint objects for real-time updates
bone_objects = {}  # Store bone objects for real-time updates

# COCO dataset skeleton structure (pairs of connected keypoints)
COCO_SKELETON = [
    ("Nose", "Left Eye"), ("Nose", "Right Eye"), ("Left Eye", "Left Ear"), ("Right Eye", "Right Ear"),
    ("Left Shoulder", "Right Shoulder"), ("Left Shoulder", "Left Elbow"), ("Right Shoulder", "Right Elbow"),
    ("Left Elbow", "Left Wrist"), ("Right Elbow", "Right Wrist"),
    ("Left Hip", "Right Hip"), ("Left Shoulder", "Left Hip"), ("Right Shoulder", "Right Hip"),
    ("Left Hip", "Left Knee"), ("Right Hip", "Right Knee"),
    ("Left Knee", "Left Ankle"), ("Right Knee", "Right Ankle")
]

class LiveKeypointUpdater(bpy.types.Operator):
    """Fetch keypoints and update scene in real-time"""
    bl_idname = "wm.live_keypoints"
    bl_label = "Live Keypoints Updater"
    _timer = None
    
    def fetch_keypoints(self):
        """Fetch keypoints from the server."""
        try:
            response = requests.get(URL)
            return response.json()
        except Exception as e:
            print(f"Error fetching keypoints: {e}")
            return []

    def create_or_update_sphere(self, name, location, radius=0.5):
        """Create a sphere if it does not exist, otherwise update its location."""
        if name in keypoint_objects:
            keypoint_objects[name].location = location
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
            obj = bpy.context.object
            obj.name = name
            keypoint_objects[name] = obj

    def create_or_update_bone(self, name, start, end):
        """Create or update a bone (cylinder) between two keypoints."""
        start_vec = mathutils.Vector(start)
        end_vec = mathutils.Vector(end)
        mid_point = (start_vec + end_vec) / 2
        direction = end_vec - start_vec
        length = direction.length

        if name in bone_objects:
            obj = bone_objects[name]
            obj.location = mid_point
            obj.scale.z = length / 2  # Adjust scale to match distance
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')
        else:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.2, location=mid_point)
            obj = bpy.context.object
            obj.name = name
            obj.scale.z = length / 2
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')
            bone_objects[name] = obj


    def remove_missing_objects(self, current_keypoints, current_bones):
        """Remove keypoints and bones that are no longer in the latest data."""
        missing_keys = set(keypoint_objects.keys()) - current_keypoints
        missing_bones = set(bone_objects.keys()) - current_bones
        
        for key in missing_keys:
            obj = keypoint_objects.pop(key, None)
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        
        for bone in missing_bones:
            obj = bone_objects.pop(bone, None)
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)

    def modal(self, context, event):
        """Continuously fetch and update keypoints while operator is running."""
        scale = 20  # Scaling factor for visualization
        if event.type == 'TIMER':
            keypoints = self.fetch_keypoints()
            if keypoints:
                keypoint_positions = {point["name"]: (point["x"] * scale, point["y"] * scale, point["z"] * scale) for point in keypoints}
                updated_keys = set(keypoint_positions.keys())
                updated_bones = set()

                for name, position in keypoint_positions.items():
                    self.create_or_update_sphere(name, position)

                for joint in COCO_SKELETON:
                    if joint[0] in keypoint_positions and joint[1] in keypoint_positions:
                        bone_name = f"{joint[0]}_{joint[1]}"
                        self.create_or_update_bone(bone_name, keypoint_positions[joint[0]], keypoint_positions[joint[1]])
                        updated_bones.add(bone_name)

                self.remove_missing_objects(updated_keys, updated_bones)
        
        return {'PASS_THROUGH'}

    def execute(self, context):
        """Start the modal timer to update keypoints live."""
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        """Stop the live update when canceled."""
        context.window_manager.event_timer_remove(self._timer)
        print("Stopped live keypoints update.")

# Register the Operator
def register():
    bpy.utils.register_class(LiveKeypointUpdater)

def unregister():
    bpy.utils.unregister_class(LiveKeypointUpdater)

if __name__ == "__main__":
    register()
    bpy.ops.wm.live_keypoints()  # Start Live Updates
