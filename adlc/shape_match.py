"""
Defines and matches contours to shapes listed in SUAS rules: circle, semicircle, quarter circle, triangle,
rectangle, pentagon, star, and cross
"""

import cv2
import numpy as np

# Radius for reference contours
R = 25

MATCH_MODE = cv2.CONTOURS_MATCH_I2

class ContourShapeMatcher:
    TRIANGLE = np.array(
        [[-((R * np.sqrt(3)) / 2), -(R / 2)], [0, R], [(R * np.sqrt(3)) / 2, -(R / 2)]]
    )

    RECTANGLE = np.array(
        [
            [-(R / np.sqrt(2)), -(R / np.sqrt(2))],
            [-(R / np.sqrt(2)), R / np.sqrt(2)],
            [R / np.sqrt(2), -(R / np.sqrt(2))],
            [R / np.sqrt(2), R / np.sqrt(2)],
        ]
    )
    PENTAGON = np.array(
        [
            [-R * np.sqrt(5 / 8 + np.sqrt(5) / 8), (R / 4) * (-1 + np.sqrt(5))],
            [-R * np.sqrt(5 / 8 - np.sqrt(5) / 8), (R / 4) * (-1 - np.sqrt(5))],
            [0, R],
            [R * np.sqrt(5 / 8 - np.sqrt(5) / 8), (R / 4) * (-1 - np.sqrt(5))],
            [R * np.sqrt(5 / 8 + np.sqrt(5) / 8), (R / 4) * (-1 + np.sqrt(5))],
        ]
    )

    # Pentagram of radius R
    # See: https://www.desmos.com/calculator/iwswcr7wy0
    A = np.pi / 2
    R_i = R * (np.sqrt(5) - 3) / 2
    STAR = np.array(
        [
            # Outer
            [-R * np.cos(A - (np.pi / 5)), -R * np.sin(A - (np.pi / 5))],
            [-R * np.sin(A - (np.pi / 10)), R * np.cos(A - (np.pi / 10))],
            [-R * np.cos(A + (np.pi / 5)), -R * np.sin(A + (np.pi / 5))],
            [R * np.cos(A), R * np.sin(A)],
            [R * np.sin(A + (np.pi / 10)), -R * np.cos(A + (np.pi / 10))],
            # Inner
            [-R_i * np.cos(A - (np.pi / 5)), -R_i * np.sin(A - (np.pi / 5))],
            [-R_i * np.sin(A - (np.pi / 10)), R_i * np.cos(A - (np.pi / 10))],
            [-R_i * np.cos(A + (np.pi / 5)), -R_i * np.sin(A + (np.pi / 5))],
            [R_i * np.cos(A), R_i * np.sin(A)],
            [R_i * np.sin(A + (np.pi / 10)), -R_i * np.cos(A + (np.pi / 10))],
        ]
    )

    CIRCLE = None
    SEMICIRCLE = None
    QUARTER_CIRCLE = None
    CROSS = None

    def __init__(self):
        """
        Initialize shapes where vertices are not explicitly defined
        """

        circle = np.full((2*R, 2*R), 255, dtype=np.uint8)
        cv2.circle(circle, (R, R), R, 255, -1)
        _, thresh = cv2.threshold(circle, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 2, 1)[-2:]
        self.CIRCLE = contours[0]

        semicircle = np.full((2*R, 2*R), 255, dtype=np.uint8)
        cv2.ellipse(semicircle, (R, R), (R, R), 0, 0, 180, 255, -1)
        _, thresh = cv2.threshold(semicircle, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 2, 1)[-2:]
        self.SEMICIRCLE = contours[0]

        quarter_circle = np.full((2*R, 2*R), 255, dtype=np.uint8)
        cv2.ellipse(quarter_circle, (R, R), (R, R), 0, 0, 90, 255, -1)
        _, thresh = cv2.threshold(quarter_circle, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 2, 1)[-2:]
        self.QUARTER_CIRCLE = contours[0]

        cross = cv2.imread("img/white_cross.png", cv2.IMREAD_GRAYSCALE)
        cross = cv2.resize(cross, (2*R, 2*R))
        _, thresh = cv2.threshold(cross, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 2, 1)[-2:]
        self.CROSS = contours[0]

    def match_contour(self, target) -> str:
        """
        Find closest matching shape name from possible shapes.

        Potentially add early stopping in future
        """

        contour_list = [
            self.TRIANGLE,
            self.RECTANGLE,
            self.PENTAGON,
            self.STAR,
            self.CIRCLE,
            self.SEMICIRCLE,
            self.QUARTER_CIRCLE,
            self.CROSS,
        ]
        contour_names = [
            "TRIANGLE",
            "RECTANGLE",
            "PENTAGON",
            "STAR",
            "CIRCLE",
            "SEMICIRCLE",
            "QUARTER_CIRCLE",
            "CROSS",
        ]

        max_val = -1
        max_idx = -1

        for i, shape in enumerate(contour_list):
            match = cv2.matchShapes(target, shape, MATCH_MODE, 0.0)
            # print(f"{contour_names[i]}: {match}")
            if match > max_val:
                max_val = match
                max_idx = i

        return contour_names[max_idx]