import numpy as np
from typing import Tuple


#  Utility for drawing circles around locations on a geodetic map
def shoot(
    lon: float, lat: float, azimuth: float, maxdist: float
) -> Tuple[float, float, float]:
    """Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq
    """
    glat1 = lat * np.pi / 180.0
    glon1 = lon * np.pi / 180.0
    s = maxdist / 1.852
    faz = azimuth * np.pi / 180.0
    EPS = 0.00000000005
    if (np.absolute(np.cos(glat1)) < EPS) and not (np.absolute(np.sin(faz)) < EPS):
        print("Only N-S courses are meaningful, starting at a pole!")
    a = 6378.13 / 1.852
    f = 1 / 298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if cf == 0:
        b = 0.0
    else:
        b = 2.0 * np.arctan2(tu, cf)

    cu = 1.0 / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1.0 + np.sqrt(1.0 + c2a * (1.0 / (r * r) - 1.0))
    x = (x - 2.0) / x
    c = 1.0 - x
    c = (x * x / 4.0 + 1.0) / c
    d = (0.375 * x * x - 1.0) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    sy = np.sin(y)
    cy = np.cos(y)
    cz = np.cos(b + y)
    e = 2.0 * cz * cz - 1.0
    while np.absolute(y - c) > EPS:
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2.0 * cz * cz - 1.0
        c = y
        x = e * cy
        y = e + e - 1.0
        y = (
            ((sy * sy * 4.0 - 3.0) * y * cz * d / 6.0 + x) * d / 4.0 - cz
        ) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2 * np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3.0 * c2a + 4.0) * f + 4.0) * c2a * f / 16.0
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1.0 - c) * d * f + np.pi) % (2 * np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180.0 / np.pi
    glat2 *= 180.0 / np.pi
    baz *= 180.0 / np.pi

    return (glon2, glat2, baz)


def get_equi_circle(
    centerlon: float, centerlat: float, radius: float, azimuths=range(0, 360)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get longitudes and latitudes corresponding to a circle centered at `centerlon, centerlat`
    """
    lons = []
    lats = []
    for azimuth in azimuths:
        glon2, glat2, baz = shoot(centerlon, centerlat, azimuth, radius)
        lons.append(glon2)
        lats.append(glat2)
    lons.append(lons[0])
    lats.append(lats[0])
    return np.array(lons), np.array(lats)
