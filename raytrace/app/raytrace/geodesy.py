import numpy as np
from numba import jit
import math

## Based on github.com/xoolive/geodesy, which bore the following license:
# The MIT License (MIT)
# 
# Copyright (c) 2015 Xavier Olive
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

@jit('f8[:],f8[:],f8[:],f8[:],f8[:],f8[:]',nopython=True,fastmath=True,error_model='numpy')
def _distance_bearing(from_lat, from_lon, to_lat, to_lon, distance, bearing):
    for i in range(from_lat.size):
        dlat = to_lat[i] - from_lat[i]
        dlon = to_lon[i] - from_lon[i]
        cos_lat1 = math.cos(from_lat[i])
        cos_lat2 = math.cos(to_lat[i])

        sin_dlat_2 = math.sin(dlat / 2.0)
        sin_dlon_2 = math.sin(dlon / 2.0)
        a = sin_dlat_2 * sin_dlat_2 + sin_dlon_2 * sin_dlon_2 * cos_lat1 * cos_lat2
        distance[i] = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        y1 = math.sin(dlon) * cos_lat1
        x1 = cos_lat1 * math.sin(to_lat[i]) - math.sin(from_lat[i]) * cos_lat2 * math.cos(dlon)

        bearing[i] = math.atan2(y1, x1) % (2 * math.pi)

def distance_bearing(from_lat, from_lon, to_lat, to_lon):
    distance = np.zeros(from_lat.size)
    bearing = np.zeros(from_lat.size)
    _distance_bearing(
        from_lat.flatten(), from_lon.flatten(),
        to_lat.flatten(), to_lon.flatten(),
        distance, bearing,
    )

    return distance.reshape(from_lat.shape), bearing.reshape(from_lat.shape)

@jit('f8[:],f8[:],f8[:],f8[:],f8[:],f8[:]',nopython=True,fastmath=True,error_model='numpy')
def _destination(from_lat, from_lon, bearing, distance, to_lat, to_lon):
    for i in range(from_lat.size):
        sin_lat1 = math.sin(from_lat[i])
        cos_lat1 = math.cos(from_lat[i])
        sin_d = math.sin(distance[i])
        cos_d = math.cos(distance[i])

        to_lat[i] = math.asin(sin_lat1 * cos_d + cos_lat1 * sin_d * math.cos(bearing[i]))
        to_lon[i] = from_lon[i] + math.atan2(math.sin(bearing[i]) * sin_d * cos_lat1, cos_d - sin_lat1 * math.sin(to_lat[i]))

def destination(from_lat, from_lon, bearing, distance):
    to_lat = np.zeros(from_lat.size)
    to_lon = np.zeros(from_lat.size)
    _destination(
        from_lat.flatten(), from_lon.flatten(),
        bearing.flatten(), distance.flatten(),
        to_lat, to_lon,
    )

    return to_lat.reshape(from_lat.shape), to_lon.reshape(from_lat.shape)


