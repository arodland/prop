import numpy as np
from math import radians

r_earth_km = 6371
r_earth_m = 1000 * r_earth_km
maxhop = 0.62784
min_phi = radians(10)
max_phi = radians(89)
m9_1 = 1.05
m9_2 = 0.989
min_takeoff = radians(1)
max_takeoff = radians(30)
sex_1 = 1.0
sex_2 = 0.965
sex_3 = 2.0
sex_4 = -0.5
ls_1 = 0.028
ls_2 = 2.0

lof_distance_base = 1.
lof_threshold = 100.
