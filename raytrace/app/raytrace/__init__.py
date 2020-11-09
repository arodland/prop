from . import constants
import geodesy.sphere
import numpy as np

r0 = constants.r_earth_km

def mof_lof(iono, from_lat, from_lon, to_lat, to_lon, longpath=False, h_min_flag=True):
    dist = geodesy.sphere.distance(from_lat.flatten(), from_lon.flatten(), to_lat.flatten(), to_lon.flatten(), radian=True).reshape(from_lat.shape) / constants.r_earth_m

    bearing = (geodesy.sphere.bearing(from_lat.flatten(), from_lon.flatten(), to_lat.flatten(), to_lon.flatten(), radian=True).reshape(from_lat.shape)) % (2 * np.pi)

    if longpath:
        dist = 2 * np.pi - dist
        bearing = (-bearing) % (2 * np.pi)

    khop = np.floor(dist / constants.maxhop).astype(int) + 1
    half_hop = dist / (2 * khop)
    max_khop = np.max(khop)

    h_avg = np.zeros_like(to_lat)
    h_min = np.full_like(to_lat, 10000.)

    for hop in range(1, max_khop+1):
        idx = hop <= khop
        hop_frac = 2*hop - 1
        cp_lat, cp_lon = geodesy.sphere.destination(from_lat[idx], from_lon[idx], bearing[idx], half_hop[idx] * hop_frac * constants.r_earth_m, radian=True)
        cp_lat = (cp_lat + np.pi / 2) % np.pi - np.pi / 2
        cp_lon = (cp_lon + np.pi) % (2 * np.pi) - np.pi
        h_min[idx] = np.fmin(h_min[idx], iono.hmf2.predict(cp_lat, cp_lon))
        h_avg[idx] += iono.hmf2.predict(cp_lat, cp_lon)

    h_avg /= khop

    h_calc = h_min if h_min_flag else h_avg

    calc_maxhop = 2 * np.arccos(r0 / (h_calc + r0))
    khop = np.floor(dist / calc_maxhop).astype(int) + 1
    half_hop = dist / (2 * khop)
    max_khop = np.max(khop)

    phi = np.arctan2(np.sin(half_hop), (((r0 + h_calc)/r0)-np.cos(half_hop)))
    phi = np.clip(phi, constants.min_phi, constants.max_phi)

    m9 = constants.m9_1 * np.power(constants.m9_2, khop) / np.cos(phi)

    takeoff = np.arctan2(np.cos(half_hop) - (r0/(h_calc+r0)), np.sin(half_hop))
    takeoff = np.clip(takeoff, constants.min_takeoff, constants.max_takeoff)

    sex = constants.sex_1 - constants.sex_2 * np.power(np.cos(takeoff), constants.sex_3)

    clof = np.zeros_like(to_lat)
    cmof = np.full_like(to_lat, 10000.)

    for hop in range(1, max_khop+1):
        idx = hop <= khop
        hop_frac = 2*hop - 1
        cp_lat, cp_lon = geodesy.sphere.destination(from_lat[idx], from_lon[idx], bearing[idx], half_hop[idx] * hop_frac * constants.r_earth_m, radian=True)
        cp_lon = (cp_lon + np.pi) % (2 * np.pi) - np.pi
        cp_lat = (cp_lat + np.pi / 2) % np.pi - np.pi / 2

        cmof[idx] = np.fmin(cmof[idx], iono.fof2.predict(cp_lat, cp_lon))
        ls = constants.ls_1 * np.power(iono.foe.predict(cp_lat, cp_lon), constants.ls_2)
        clof[idx] += ls

    ### TODO: TEP

    cmof *= m9
    cmof = np.clip(cmof, 1.0, 50.0)

    clof *= np.power(np.abs(sex), constants.sex_4)
    clof = np.clip(clof, 1.0, 50.0)

    return cmof, clof
