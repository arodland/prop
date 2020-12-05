from . import constants
from . import geodesy
import numpy as np

r0 = constants.r_earth_km

def mof_lof(iono, from_lat, from_lon, to_lat, to_lon, longpath=False, h_min_flag=True):
    dist, bearing = geodesy.distance_bearing(from_lat, from_lon, to_lat, to_lon)

    if longpath:
        dist = 2 * np.pi - dist
        bearing = (bearing + np.pi) % (2 * np.pi)

    khop = np.floor(dist / constants.maxhop).astype(int) + 1
    half_hop = dist / (2 * khop)
    max_khop = np.max(khop)

    h_avg = np.zeros_like(to_lat)
    h_min = np.full_like(to_lat, 10000.)

    for hop in range(1, max_khop+1):
        idx = hop <= khop
        hop_frac = 2*hop - 1
        cp_lat, cp_lon = geodesy.destination(from_lat[idx], from_lon[idx], bearing[idx], half_hop[idx] * hop_frac)
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

    phi = np.arctan2(np.sin(half_hop), (1.0 + h_calc/r0 - np.cos(half_hop)))
    phi = np.clip(phi, constants.min_phi, constants.max_phi)

    m9 = constants.m9_1 * np.power(constants.m9_2, khop) / np.cos(phi)

    takeoff = np.arctan2(np.cos(half_hop) - (r0/(h_calc+r0)), np.sin(half_hop))
    takeoff_true = takeoff
    takeoff = np.clip(takeoff, constants.min_takeoff, constants.max_takeoff)

    cmof = np.full_like(to_lat, 10000.)

    rms_gyf = np.zeros_like(to_lat)
    sum_I = np.zeros_like(to_lat)

    for hop in range(1, max_khop+1):
        idx = hop <= khop
        hop_frac = 2*hop - 1
        cp_lat, cp_lon = geodesy.destination(from_lat[idx], from_lon[idx], bearing[idx], half_hop[idx] * hop_frac)
        cp_lon = (cp_lon + np.pi) % (2 * np.pi) - np.pi
        cp_lat = (cp_lat + np.pi / 2) % np.pi - np.pi / 2
        cmof[idx] = np.fmin(cmof[idx], iono.fof2.predict(cp_lat, cp_lon))

        sum_I[idx] += np.clip(np.exp(0.8445 * iono.foe.predict(cp_lat, cp_lon) - 2.937) - 0.4, 0, None)
        rms_gyf[idx] += np.power(iono.gyf.predict(cp_lat, cp_lon), 2)

    rms_gyf = np.sqrt(rms_gyf / khop)

    ### TODO: TEP

    cmof *= m9
    cmof = np.clip(cmof, 1.0, 50.0)

    pathlen = 2 * khop * (h_calc + r0 * (1 - np.cos(half_hop))) / np.sin(half_hop + takeoff_true)
    g_loss = np.clip(20 * np.log10(pathlen / constants.lof_distance_base), 0.0, None)

    # https://apps.dtic.mil/dtic/tr/fulltext/u2/a269557.pdf

    loss_tgt = constants.lof_threshold - g_loss
    loss_sec = loss_tgt * np.cos(phi)

    clof = np.full_like(to_lat, 2.0)
    clof = np.power(np.clip(677.2 * sum_I / loss_sec - 10.2, 0.0, None), 1 / 1.98) - rms_gyf
    # Empirical correction for the fact that the above calculation is missing the 20*log10(f) loss
    clof += 0.0591 - 4.344 * sum_I * (np.exp(-0.0775 * loss_sec) + 0.00366 * np.sqrt(loss_sec))
    clof = np.clip(clof, 1.0, None)

    return {
        'mof': cmof,
        'lof': clof,
        'm9': m9,
        'takeoff': takeoff,
        'phi': phi,
        'khop': khop.astype(float),
        'half_hop': half_hop,
        'pathlen': pathlen,
        'distance': dist,
        'bearing': bearing,
        'g_loss': g_loss,
        'rms_gyf': rms_gyf,
        'sum_I': 0.0 + sum_I.astype(float),
    }
