"""
Defines and matches contours to shapes listed in SUAS rules: circle, semicircle, quarter circle, triangle,
rectangle, pentagon, star, and cross
"""

import cv2
import numpy as np

X = 50
MATCH_MODE = cv2.CONTOURS_MATCH_I2

TRIANGLE = np.array([[0, X], [X, X], [25, np.sqrt(2)*X]])
RECTANGLE = np.array([[0, 0], [0, X], [X,0], [X,X]]) 
PENTAGON = np.array([[25,0],[1,17],[10,45],[40,45],[49,17]])

CIRCLE = None
SEMICIRCLE = None
QUARTER_CIRCLE = None
CROSS = None
STAR = None

def init_shapes():
    circle = np.zeros((50,50))
    cv2.circle(circle, (25,25),25,255,-1)
    ret, thresh = cv2.threshold(circle, 127, 255,0)
    _,contours,hierarchy = cv2.findContours(thresh,2,1)
    CIRCLE = contours[0]
    
    semicircle = np.zeros((50,50))
    cv2.ellipse(semicircle,(25,25),(25,25),0,0,180,255,-1)
    ret, thresh = cv2.threshold(semicircle, 127, 255,0)
    _,contours,hierarchy = cv2.findContours(thresh,2,1)
    SEMICIRCLE = contours[0]

    quarter_circle = np.zeros((50,50))
    cv2.ellipse(quarter_circle,(25,25),(25,25),0,0,90,255,-1)
    ret, thresh = cv2.threshold(quarter_circle, 127, 255,0)
    _,contours,hierarchy = cv2.findContours(thresh,2,1)
    QUARTER_CIRCLE = contours[0]

    cross = cv2.imread('img/white_cross.png', cv2.IMREAD_GRAYSCALE)
    cross = cv2.resize(cross, (50,50))
    ret, thresh = cv2.threshold(cross, 127, 255,0)
    _,contours,hierarchy = cv2.findContours(thresh,2,1)
    CROSS = contours[0]

    star = cv2.imread('img/white_star.png', cv2.IMREAD_GRAYSCALE)
    star = cv2.resize(star, (50,50))
    ret, thresh = cv2.threshold(star, 127, 255,0)
    _,contours,hierarchy = cv2.findContours(thresh,2,1)
    STAR = contours[0]

def match_contour(target) -> str:
    """
    Find closest matching shape name from possible shapes.

    Potentially add early stopping in future
    """

    contour_list  =  [TRIANGLE,RECTANGLE,PENTAGON,CIRCLE,SEMICIRCLE,QUARTER_CIRCLE,CROSS,STAR]
    contour_names =  ["TRIANGLE","RECTANGLE","PENTAGON","CIRCLE","SEMICIRCLE","QUARTER_CIRCLE","CROSS","STAR"]

    max_val = -1
    max_idx = -1

    for i, shape in enumerate(contour_list):
        match = cv2.matchShapes(target, shape, MATCH_MODE, 0.0)
        print(f"{contour_names[i]}: {match}")
        if match > max_val:
            max_val = match
            max_idx = i
    
    return contour_names[max_idx]



if __name__ == "__main__":
    # Debug only
    init_shapes()

