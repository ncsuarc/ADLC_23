"""
Defines and matches contours to shapes listed in SUAS rules: circle, semicircle, quarter circle, triangle,
rectangle, pentagon, star, and cross
"""

import time
import cv2
import numpy as np

# Radius for reference contours
R = 25

MATCH_MODE = cv2.CONTOURS_MATCH_I3


class ContourShapeMatcher:
    TRIANGLE = np.add(np.array(
        [
            [[-((R * np.sqrt(3)) / 2), -(R / 2)]],
            [[0, R]],
            [[(R * np.sqrt(3)) / 2, -(R / 2)]],
        ],
        dtype=np.int32,
    ),R)

    RECTANGLE = np.add(np.array(
        [
            [[R / np.sqrt(2), -(R / np.sqrt(2))]],
            [[-(R / np.sqrt(2)), -(R / np.sqrt(2))]],
            [[-(R / np.sqrt(2)), R / np.sqrt(2)]],
            [[R / np.sqrt(2), R / np.sqrt(2)]],
        ],
        dtype=np.int32,
    ),R)

    PENTAGON = np.add(np.array(
        [
            [[R * np.sqrt(5 / 8 + np.sqrt(5) / 8), (R / 4) * (-1 + np.sqrt(5))]],
            [[0, R]],
            [[-R * np.sqrt(5 / 8 + np.sqrt(5) / 8), (R / 4) * (-1 + np.sqrt(5))]],
            [[-R * np.sqrt(5 / 8 - np.sqrt(5) / 8), (R / 4) * (-1 - np.sqrt(5))]],
            [[R * np.sqrt(5 / 8 - np.sqrt(5) / 8), (R / 4) * (-1 - np.sqrt(5))]],
        ],
        dtype=np.int32,
    ),R)

    # Pentagram of radius R
    # See: https://www.desmos.com/calculator/iwswcr7wy0
    A = np.pi / 2
    R_i = R * (np.sqrt(5) - 3) / 2
    STAR = np.add(np.array(
        [
            [[R * np.cos(A), R * np.sin(A)]],
            [[-R_i * np.cos(A - (np.pi / 5)), -R_i * np.sin(A - (np.pi / 5))]],
            [[R * np.sin(A + (np.pi / 10)), -R * np.cos(A + (np.pi / 10))]],
            [[-R_i * np.sin(A - (np.pi / 10)), R_i * np.cos(A - (np.pi / 10))]],
            [[-R * np.cos(A + (np.pi / 5)), -R * np.sin(A + (np.pi / 5))]],
            [[R_i * np.cos(A), R_i * np.sin(A)]],
            [[-R * np.cos(A - (np.pi / 5)), -R * np.sin(A - (np.pi / 5))]],
            [[R_i * np.sin(A + (np.pi / 10)), -R_i * np.cos(A + (np.pi / 10))]],
            [[-R * np.sin(A - (np.pi / 10)), R * np.cos(A - (np.pi / 10))]],
            [[-R_i * np.cos(A + (np.pi / 5)), -R_i * np.sin(A + (np.pi / 5))]],
        ],
        dtype=np.int32,
    ),R)

    CIRCLE = None
    SEMICIRCLE = None
    QUARTER_CIRCLE = None
    CROSS = None

    def __init__(self):
        """
        Initialize shapes where vertices are not explicitly defined
        """

        circle = np.full((2 * R, 2 * R), 0, dtype=np.uint8)
        cv2.circle(circle, (R, R), R, 255, -1)
        _, thresh = cv2.threshold(circle, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 2, 1)[-2:]
        self.CIRCLE = contours[0]

        semicircle = np.full((2 * R, 2 * R), 0, dtype=np.uint8)
        cv2.ellipse(semicircle, (R, R), (R, R), 0, 0, 180, 255, -1)
        _, thresh = cv2.threshold(semicircle, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 2, 1)[-2:]
        self.SEMICIRCLE = contours[0]

        quarter_circle = np.full((2 * R, 2 * R), 0, dtype=np.uint8)
        cv2.ellipse(quarter_circle, (R, R), (R, R), 0, 0, 90, 255, -1)
        _, thresh = cv2.threshold(quarter_circle, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 2, 1)[-2:]
        self.QUARTER_CIRCLE = contours[0]
        # cv2.drawContours(thresh, contours, -1, 127, 1)
        # cv2.imwrite("./tmp/quarter_thresh.png", thresh)

        cross = cv2.imread("img/plusplus_inv.png", cv2.IMREAD_GRAYSCALE)
        cross = cv2.resize(cross, (2 * R, 2 * R))
        _, thresh = cv2.threshold(cross, 127, 255, 0)

        contours, _ = cv2.findContours(thresh, 2, 1)[-2:]
        self.CROSS = contours[0]

    def match_contour(self, target) -> str:
        """
        Find closest matching shape name from possible shapes.

        Potentially add early stopping in future
        """

        contour_list = [
            # self.TRIANGLE,
            # self.RECTANGLE,
            # self.PENTAGON,
            # self.STAR,
            self.CIRCLE,
            self.SEMICIRCLE,
            self.QUARTER_CIRCLE,
            self.CROSS,
        ]
        contour_names = [
            # "TRIANGLE",
            # "RECTANGLE",
            # "PENTAGON",
            # "STAR",
            "CIRCLE",
            "SEMICIRCLE",
            "QUARTER_CIRCLE",
            "CROSS",
        ]

        min_val = np.Inf
        min_idx = -1

        # Based on: https://medium.com/p/bad67c40174
        # Check for simpler polygons
        approx = cv2.approxPolyDP(target, 0.01* cv2.arcLength(target, True), True)
        match len(approx):
            case 3:
                return "TRIANGLE"
            case 4:
                return "RECTANGLE"
            case 5:
                return "PENTAGON"
            case 10:
                return "STAR"


        # Switch to contour shape matching
        print(f"target: {np.shape(target)}, approx: {len(approx)}")
        for i, shape in enumerate(contour_list):
            # print(f"{contour_names[i]}: {np.shape(shape)}")
            match = cv2.matchShapes(target, shape, MATCH_MODE, 0.0)
            # print(f"target: {target}\n---")
            print(f"{contour_names[i]}: {match}")

            if match <= min_val:
                min_val = match
                min_idx = i

        # DEBUG
        # img = np.full((2*R, 2*R,3), 0, dtype=np.uint8)
        # cv2.drawContours(img, target.astype(np.int32), -1, (0,0,255), 1)
        # cv2.drawContours(img, [approx.astype(np.int32)], 0, (255,0,0), 1)
        # cv2.drawContours(img, [contour_list[min_idx]], 0, (0,255,0), 1)
        # cv2.imwrite(f"./tmp/test-{time.time()}.png", img)

        # print(contour_list[min_idx])

        print("===")

        return contour_names[min_idx]
