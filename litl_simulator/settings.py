"""Settings for simulation."""
import os
import socket


# CAMTYPES
DEPTH_CAMTYPE = "depth"
DIFFUSE_COLOR_CAMTYPE = "diffuse_color"
OPTICAL_FLOW_CAMTYPE = "optical_flow"
SPECULAR_METALLIC_ROUGHNESS_CAMTYPE = "specular_metallic_roughness"
RGB_CAMTYPE = "rgb"

# CAMS PATHS
CAM_FILENAME = "camera_data.npz"
CAM_MATRIX_FILENAME = "camera_inverse_matrix.npz"
PROJECTION_MATRIX_FILENAME = "pc2cam_proj_matrix.npz"

# PC PATHS
PC_FILENAME = "lidar_pc.npz"  # contains the OG point cloud with all data
PC_MATRIX_FILENAME = "lidar_matrix.npz"
TAGS_FILENAME = "tags.npz"    # contains tag ids for corresponding point cloud

# bboxes paths
BBOX_YAW_PATH = "cam_bboxes"
BBOX_3D_EDGES_FILENAME = "bboxes_3D_edges.npz"
BBOX_3D_IDS_FILENAME = "bboxes_3D_ids.npz"
BBOX_2D_EDGES_FILENAME = "bboxes_2D_edges.npz"
BBOX_3D_VERTICES_FILENAME = "bboxes_3D_vertices.npz"
BBOX_2D_IDS_FILENAME = "bboxes_2D_ids.npz"
BBOX_KITTI_FORMAT_FILENAME = "bboxes_kitti_format.txt"
BBOX_VERTICES_OCCLUSIONS_FILENAME = "bboxes_occlusions.npz"

# Annotations colors / types
# ANNOTATION_TAG_2_CARLA_LABEL = {
#         "TrafficSign": carla.CityObjectLabel.TrafficSigns,
#         "TrafficLight": carla.CityObjectLabel.TrafficLight,
#         "Car": carla.CityObjectLabel.Vehicles,
#         "Pedestrian": carla.CityObjectLabel.Pedestrians,
#         }
KITTI_TYPES = (
            'Car', 'Van', 'TrafficSign', 'TrafficLight', 'Truck',
            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
            'Misc', 'DontCare',
            )
KITTI_TYPE_2_BBOX_COLOR = {
        "TrafficSign": [235 / 255, 235 / 255, 52 / 255, 1],  # yellow
        "TrafficLight": [52 / 255, 235 / 255, 216 / 255, 1],  # teal
        "Car": [1, 0, 0, 1],  # red
        "Van": [118 / 255, 68 / 255, 138 / 255, 1],  # purple
        "Truck": [120 / 255, 40 / 255, 31 / 255, 1],  # dark red
        "Cyclist": [1, 95 / 255, 114 / 255, 1],  # orange
        "Pedestrian": [55 / 255, 235 / 255, 52 / 255, 1],  # green
        "Person_sitting": [55 / 255, 235 / 255, 52 / 255, 1],  # green
        }
# occlusion/bbox flags
OK_FLAG = 2
OUTSIDE_CAMERA_FOV_FLAG = 1
OCCLUDED_FLAG = 0
# MIN_VISIBLE_VERTICES_FOR_RENDER = 4
MIN_BBOX_AREA_IN_PX = 100

# subfolder basename
ANNOTATIONS_FOLDER = "annotations"
RAW_DATA_FOLDER = "raw_data"
YAW_BASENAME = "yaw"

# Processed data paths
METADATA_FILENAME = "process_metadata.json"
MOVIE_FRAMES_DIRECTORY = "movie_frames"
PRESET_SUBFOLDER_BASENAME = "preset"
PROCESSED_DATA_FOLDER = "processed_data"
PROCESSED_CAM_FILENAME = CAM_FILENAME.replace(".npz", ".png")
PROCESSED_COLORS_FILENAME = "processed_colors.npy"
PROCESSED_INTENSITIES_FILENAME = "processed_intensities.npy"
PROCESSED_PC_FILENAME = "processed_pc.npy"  # non-compressed for faster reading

USE_GPU = True
# (over npts, over waveform resolution)
# GPU_THREADS_PER_BLOCK = (25, 24)  # for npts=875
GPU_THREADS_PER_BLOCK = (5, 24)  # for fraction=0.15 -> npts=131

# Viewer
WHITE_BACKGROUND = True
SHOW_ONLY_CAR_PED_CYC = True
SCREENSHOT_SAVE_PATH = '/lhome/dscheub/ObjectDetection/data/external/CarlaPaper'

if socket.gethostname() == 'lreliemp055':
    # Felix's laptop
    SCREENSHOT_SAVE_PATH = os.path.expanduser("~/Workspace/lidarsim_data/screenshots")
