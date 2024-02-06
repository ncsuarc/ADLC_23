"""
Test of depth and geolocation in image as described in:

S. Sanyal, S. Bhushan and K. Sivayazi, "Detection and Location Estimation of Object in 
Unmanned Aerial Vehicle using Single Camera and GPS," 2020 First International 
Conference on Power, Control and Computing Technologies (ICPC2T), Raipur, India, 2020, 
pp. 73-78, doi: 10.1109/ICPC2T48082.2020.9071439.

CURRENTLY NOT FUNCTIONAL
"""

import numpy as np
from PIL.Image import ExifTags, Exif

## Camera Calibration #############
# TODO: Actually calibrate camera (camera_calibration package)
# Using data for Hero3 black () for now

# Focal length mm?
OP = 24.4
# Focal length in pixels (same for x,y?)
F = 2.6126803987523554  # not accurate
F_X = F
F_Y = F
# width in pixels
W = 5568
# height
H = 4872
# center x
C_X = 2784
# center y
C_Y = 2436
# Skew (probably 0)
S = 0


def _extract_attitude(xmp_data):
    pitch = xmp_data.ncsu.ATTITUDE.pitch
    roll = xmp_data.ncsu.ATTITUDE.roll
    yaw = xmp_data.ncsu.ATTITUDE.yaw

    heading = 0 # TODO: calc from attitude
    angle = 0 # TODO: calc from attitude

    alt = xmp_data.ncsu.GLOBAL_POSITION_INT.relative_alt / 10e2
    long = xmp_data.ncsu.GLOBAL_POSITION_INT.lon / 10e6
    lat = xmp_data.ncsu.GLOBAL_POSITION_INT.lat / 10e6

    return alt, angle, long, lat, heading


def geolocate(x_t, y_t, xmp_data: Exif) -> tuple[int, int]:
    cam_height, cam_angle, cam_long, cam_lat, heading = _extract_attitude(
        xmp_data
    )

    ## Depth estimation ################

    # Normalize
    x_n = (x_t - C_X) / F_X
    y_n = (y_t - C_Y) / F_Y

    # principle point depth
    z_p = cam_height * (1 / np.cos(cam_angle))

    # distances from principle points
    f = 50

    # dist from principle to vertical
    ap = H / 2
    # dist from principle to horizontal
    bp = W / 2

    # eqn 6 inserted to 8
    d_x = 1000 * f * (bp / OP)
    # eqn 7 inserted to 9
    d_y = 1000 * f * (ap / OP)

    # Scaling factors
    s_x = C_X / d_x
    s_y = C_Y / d_y

    # pf and angle of F
    # `f` changes to mm now?????
    d = np.sqrt(((C_X - x_n) / s_x) ** 2 + ((C_Y - y_n) / s_y) ** 2)
    theta = np.arctan(d / f)

    # Distance between camera and object
    z = z_p * (1 / np.cos(theta))

    # Camera frame coords
    x_c = x_n * z
    y_c = y_n * z
    z_c = z

    # Body frame coords
    x_b = x_c
    y_b = y_c * np.cos(cam_angle) + z_c * np.sin(cam_angle)
    z_b = -y_c * np.sin(cam_angle) + z_c * np.cos(cam_angle)

    # Convert to ENU coords
    E = z_b * np.sin(heading) + x_b * np.cos(heading)
    N = z_b * np.cos(heading) - x_b * np.sin(heading)
    U = -y_b

    ## GPS Location estimation ################

    # bearing of target point
    b = np.arctan(np.abs(E / N))

    # distance to point
    s = np.sqrt(E**2 + N**2)

    # corrections
    dX = s * np.sin(b)
    dY = s * np.cos(b)

    # lat/long corrections
    d_long = dX / (11320 * np.cos(cam_lat))
    d_lat = dY / 110540

    # final
    target_long = cam_long + d_long
    target_lat = cam_lat + d_lat

    return target_long, target_lat
