import ctypes
import os

# Load the librealsense2 shared library
# Adjust the path to the shared library file (.so file on Linux)
librealsense = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/librealsense2.so")

# Example: Retrieve librealsense version
librealsense.rs2_get_version.restype = ctypes.c_char_p  # Define return type
version = librealsense.rs2_get_version()
print(f"Librealsense version: {version.decode('utf-8')}")
