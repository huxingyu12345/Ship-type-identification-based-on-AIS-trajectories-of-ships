import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    """两点球面距离，单位公里。"""
    radius = 6371
    lat1, lon1, lat2, lon2 = map(np.array, [lat1, lon1, lat2, lon2])
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
