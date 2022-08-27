from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents bonding box around the single house number. Bounding box is
    a rectangle described by two dots: upper left and lower right.

    Parameters
    ----------
    x1 : int
        x coordinate of the upper left dot
    y1 : int
        y coordinate of the upper left dot
    x2 : int
        x coordinate of the lower right dot
    y2 : int
        y coordinate of the lower left dot
    """
    x1: int
    y1: int
    x2: int
    y2: int

    def transform(
        self,
        ratio: float,
        top: int,
        bottom: int,
        left: int,
        right: int,
    ) -> None:
        """Transforms current bounding box after image was resized.

        Parameters
        ----------
        ratio : float
            aspect ratio that has to be retained from the initial image
        top : int
            how many pixels add to the top
        bottom : int
            how many pixels add to the bottom
        left : int
            how many pixels add to the left
        right : int
            how many pixels add to the right
        """
        self.x1 = int(self.x1 * ratio + left)
        self.y1 = int(self.y1 * ratio + top)
        self.x2 = int(self.x2 * ratio + right)
        self.y2 = int(self.y2 * ratio + bottom)
