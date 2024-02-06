"""
Simple color to text using the color names listed in the SUAS 2024 competition rules. More colors can be added as needed.

Since the size is limited, simply iterates over RGB vectors and selects minimum distance. A more efficient implementation would use something like a Ball Tree or a KDTree (sklearn), but until we have a latency issue this decreases dependencies and complexity.

Uses Euclidean distance between RGB vectors, but a more robust color difference metric may be needed.
"""

from math import sqrt
from sys import maxint

COLOR_MAP = [
    ("WHITE", (255, 255, 255)),
    ("BLACK", (0, 0, 0)),
    ("RED", (255, 0, 0)),
    ("BLUE", (0, 0, 255)),
    ("GREEN", (0, 255, 0)),
    ("PURPLE", (255, 0, 255)),
    ("BROWN", (123, 63, 0)),  # RGB from HTML 'chocolate'
    ("ORANGE", (255, 165, 0)),
]


def color_diff(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """
    Calculate Euclidean dist
    """
    return sqrt(sum([(x - y) ** 2 for x, y in zip(c1, c2)]))


def rgb_to_text(r_g_b: tuple[int, int, int]) -> str:
    """
    Finds color in COLOR_MAP with least difference to r_g_b
    """
    min_diff = maxint
    min_name = None
    for name, name_rgb in COLOR_MAP:
        diff = color_diff(r_g_b, name_rgb)
        if diff < min_diff:
            min_diff = diff
            min_name = name

    return min_name
