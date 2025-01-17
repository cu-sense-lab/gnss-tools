"""
Defines helpful CZML utility functions for creating packets for radio occultation etc.

Author: Brian Breitsch
Date: 2025-01-02
"""

from datetime import datetime, timedelta
import copy
from typing import Any, Dict, List, Tuple
import numpy as np


def timestring(dt: datetime) -> str:
    """Given datetime object, returns CZML string format of
    datetime."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


DOCUMENT_TEMPLATE = {
    "id": "document",
    "version": "1.0",
    "clock": {
        "interval": "{0}/{1}",
        "currentTime": "{0}",
        "multiplier": 10,
        "range": "LOOP_STOP",
        "step": "SYSTEM_CLOCK_MULTIPLIER",
    },
}


def create_document_header(start_dt: datetime, end_dt: datetime) -> Dict[str, Any]:
    """
    Create CZML document header.  The header must specify the window over which
    the visualization takes place. `window` -- 2-tuple of datetime objects
    specifying the start and end times of the visualization Returns: dict
    """
    packet = copy.deepcopy(DOCUMENT_TEMPLATE)
    start_epoch = timestring(start_dt)
    end_epoch = timestring(end_dt)
    packet["clock"]["currentTime"] = start_epoch
    packet["clock"]["interval"] = start_epoch + "/" + end_epoch
    return packet


STATIC_POINT_TEMPLATE = {
    "id": "",
    "name": "",
    "label": {
        "horizontalOrigin": "CENTER",
        "outlineColor": {"rgba": [30, 144, 255, 255]},
        "outlineWidth": 5,
        "text": "",
        "verticalOrigin": "BOTTOM",
    },
    "point": {
        "color": {"rgba": [255, 144, 255, 255]},
        "outlineColor": {"rgba": [0, 0, 0, 255]},
        "outlineWidth": 2,
        "pixelSize": 10,
    },
    "position": {"cartesian": [], "referenceFrame": "FIXED"},
}

SIMPLE_STATIC_POINT_TEMPLATE = {
    "id": "",
    "name": "",
    "point": {"color": {"rgba": [0, 0, 0, 0]}, "pixelSize": 10},
    "outlineWidth": 0,
    "position": {"cartesian": [], "referenceFrame": "FIXED"},
}


# def object_id(object_class: str, object_name: str) -> str:
#     """Get CZML object unique ID path from object class and name"""
#     return "/{0}/{1}".format(object_class, object_name)


def create_static_point_packet(
    position: Tuple[float] | np.ndarray,
    object_id: str,
    label: str = "",
    reference_frame: str = "FIXED",
    color: Tuple[float] = (1, 1, 1, 1),
    template: str = STATIC_POINT_TEMPLATE,
) -> Dict[str, Any]:
    """
    Create CZML object to represent static (non-moving relative to either ECF or ECI coord. system).
    `position` -- ndarray of shape (3,)
    `reference_frame` -- "INERTIAL" or "FIXED"
    `color` -- point color in fractional units.  (Note, this function converts to 0-255 for use by Cesium).
    Returns: dict
    """
    packet = copy.deepcopy(template)
    packet["id"] = object_id
    packet["name"] = object_id
    packet["label"]["text"] = label
    packet["position"]["referenceFrame"] = reference_frame
    position = position.astype(float)
    packet["position"]["cartesian"] = [position[0], position[1], position[2]]
    packet["point"]["color"]["rgba"] = [
        int(255 * color[0]),
        int(255 * color[1]),
        int(255 * color[2]),
        255 if len(color) < 4 else int(color[3] * 255),
    ]
    return packet


DYNAMIC_POINT_TEMPLATE = {
    "id": "",
    "name": "",
    "label": {
        "horizontalOrigin": "CENTER",
        "outlineColor": {"rgba": [30, 144, 255, 255]},
        "outlineWidth": 5,
        "text": "",
        "verticalOrigin": "BOTTOM",
    },
    "path": {
        "material": {"solidColor": {"color": {"rgba": [30, 144, 255, 255]}}},
        "leadTime": 60,
        "trailTime": 180,
    },
    "point": {
        "color": {"rgba": [30, 144, 255, 255]},
        "outlineColor": {"rgba": [0, 0, 128, 255]},
        "outlineWidth": 2,
        "pixelSize": 10,
    },
    "position": {
        "interpolationAlgorithm": "LAGRANGE",
        "interpolationDegree": 5,
        "epoch": "{0}",
        "cartesian": [],
    },
}


def create_dynamic_point_packet(
    times: List[float],
    positions: List[Tuple[float]] | np.ndarray,
    object_id: str,
    label: str = "",
    reference_frame: str = "FIXED",
) -> Dict[str, Any]:
    """
    Create CZML object to represent static (non-moving relative to either ECF or ECI coord. system).
    `position` -- ndarray of shape (3,)
    `reference_frame` -- "INERTIAL" or "FIXED"
    Returns: dict
    """
    packet = copy.deepcopy(DYNAMIC_POINT_TEMPLATE)
    packet["id"] = object_id
    packet["name"] = object_id
    packet["label"]["text"] = label
    packet["position"]["referenceFrame"] = reference_frame
    epoch, positions_list = create_positions_list(times, positions)
    packet["position"]["epoch"] = timestring(epoch)
    packet["position"]["cartesian"] = positions_list
    return packet


def create_positions_list(
    times: List[datetime], positions: List[Tuple[float]] | np.ndarray
) -> Tuple[datetime, List[float]]:
    """
    This function accepts Python list structures or and ndarray
    times - list of N datetime objects
    positions - list of N length-3 lists of point positions

    Converts (N, 3) positions list to (N, 4) time offset + positions list

    Returns: `epoch, positions_list`
    """
    N = len(times)
    # print(positions.shape)
    if isinstance(positions, np.ndarray):
        assert positions.shape == (N, 3)
        positions = positions.tolist()
    assert len(positions) == N
    epoch = times[0]
    # IMPORTANT NOTE:
    #  A CZML positions array is one long array, not an array of arrays...
    positions = [
        val
        for t, pos in zip(times, positions)
        for val in [(t - epoch).total_seconds()] + list(pos)
    ]
    return epoch, positions


RAY_TEMPLATE = {
    "id": "",
    "polyline": {
        "show": [{"boolean": False}, {"interval": "{0}/{1}", "boolean": True}],
        "width": 2,
        "material": {"solidColor": {"color": {"rgba": [180, 160, 100, 255]}}},
        "followSurface": False,
        "positions": {"references": ["{0}#position", "{1}#position"]},
    },
}


def create_ray_packet(
    point_1_id: str,
    point_2_id: str,
    window: Tuple[datetime, datetime],
    object_id: str,
    label: str = "",
    color: Tuple[int] = (218, 165, 32, 255),
) -> Dict[str, Any]:
    """
    Create CZML object to represent a line segment between two CZML point objects.
    `point_1_id`, `point_2_id` -- CZML object IDs of the two points to connect
    `window` -- 2-tuple of datetime objects over which ray is visible
    Returns: dict
    """
    packet = copy.deepcopy(RAY_TEMPLATE)
    packet["id"] = object_id
    packet["name"] = object_id
    start = timestring(window[0])
    end = timestring(window[1])
    packet["polyline"]["show"][1]["interval"] = start + "/" + end
    packet["polyline"]["positions"]["references"] = [
        point_1_id + "#position",
        point_2_id + "#position",
    ]
    packet["polyline"]["material"]["solidColor"]["color"]["rgba"] = list(color)
    return packet


PROFILE_TEMPLATE = {
    "id": "",
    "name": "",
    "label": {
        "horizontalOrigin": "CENTER",
        "outlineColor": {"rgba": [30, 144, 255, 255]},
        "outlineWidth": 5,
        "text": "",
        "verticalOrigin": "BOTTOM",
    },
    "path": {
        "material": {"solidColor": {"color": {"rgba": [30, 144, 255, 255]}}},
        "leadTime": 60,
        "trailTime": 60,
    },
    "point": {
        "color": {"epoch": "", "rgba": [0, 30, 144, 255, 255, 1, 30, 123, 123, 255]},
        "outlineColor": {"rgba": [0, 0, 128, 255]},
        "outlineWidth": 2,
        "pixelSize": 20,
    },
    "position": {
        "interpolationAlgorithm": "LAGRANGE",
        "interpolationDegree": 2,
        "epoch": "{0}",
        "cartesian": [],
    },
}

INTERSECT_TEMPLATE = {
    "id": "",
    "name": "",
    "label": {
        "horizontalOrigin": "CENTER",
        "outlineColor": {"rgba": [30, 144, 255, 255]},
        "outlineWidth": 5,
        "text": "",
        "verticalOrigin": "BOTTOM",
    },
    "path": {
        "material": {"solidColor": {"color": {"rgba": [30, 144, 255, 255]}}},
        "leadTime": 1,
        "trailTime": 1,
    },
    "point": {
        "color": {"rgba": [30, 144, 255, 255]},
        "outlineColor": {"rgba": [0, 0, 128, 255]},
        "outlineWidth": 2,
        "pixelSize": 10,
    },
    "position": {
        "interpolationAlgorithm": "LAGRANGE",
        "interpolationDegree": 5,
        "epoch": "{0}",
        "cartesian": [],
    },
}
