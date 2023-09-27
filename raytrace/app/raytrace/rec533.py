from . import constants
from . import geodesy
import numpy as np

r0 = constants.r_earth_km
km = np.array(1 / r0) # one km in radians
min_elev = np.radians(3)
emax = 2 * np.arccos(r0 / (110 + r0))

def calc_cp(iono, lat, lon):
    fof2 = iono.fof2.predict(lat, lon)
    foe = iono.foe.predict(lat, lon)
    mufd = iono.mufd.predict(lat, lon)
    gyf = iono.gyf.predict(lat, lon)
    M = mufd / fof2
    x = np.clip(fof2 / foe, 2, None)
    B = M - 0.124 * (M**2 - 4) * (0.0215 + 0.005 * np.sin(7.854 / x - 1.9635))
    dmax = 4780 + (12610 + 2140/x**2 - 49720/x**4 + 688900/x**6) * (1 / B - 0.303)
    hr = np.clip(1490 / M - 176, 0, 500)

    return {
        'fof2': fof2,
        'foe': foe,
        'gyf': gyf,
        'M': M,
        'dmax': dmax / r0,
        'hr': hr,
        'B': B,
    }

def elevation_angle(hop, hr):
    return np.arctan( 1. / np.tan(hop / 2.) - (r0 / (hr + r0)) / np.sin(hop / 2.))

def incidence_angle(takeoff, hr):
    return np.arcsin(r0 * np.cos(takeoff) / (r0 + hr))

def sun_decl(ts):
    doy = ts.timetuple().tm_yday
    return np.radians(23.45) * np.cos((doy - 172) * 2. * np.pi / 365.)

def hour_angle(ts, lon):
    hour = ts.hour + ts.minute / 60.
    return ((hour / 12. - 1) * np.pi + lon)

def cos_chi(ts, lat, lon):
    return np.clip(
        np.sin(lat) * np.sin(sun_decl(ts)) + np.cos(lat) * np.cos(sun_decl(ts)) * np.cos(hour_angle(ts, lon)),
        0.0, 1.0
    )

def C(d, dmax):
    Z = 1 - 2 * d / dmax
    return np.polyval((0.096, 0.181, 0.088, -0.090, -0.424, -0.591, 0.74), Z)

def Fd(dm_rad):
    dm = dm_rad * r0
    c = [29.1996868566837e-6, 87.4376851991085e-9, 22.0776941764705e-12, 102.342990689362e-15, -92.4986988833091e-18, 25.8520201885984e-21, -2.40074637494790e-24]
    return ((((((c[6]*dm + c[5])*dm + c[4])*dm + c[3])*dm + c[2])*dm + c[1])*dm + c[0])*dm


def nEMUF(iono, dist, mode, info):
    hop = dist / mode

    takeoff = elevation_angle(hop, 110.)
    i110 = incidence_angle(takeoff, 110.)
    return info['foe'] / np.cos(i110)

def nF2MUF(iono, dist, mode, info, clip_dmax=True):
    dmax = np.clip(info['dmax'], 0, 4000*km) if clip_dmax else info['dmax']
    d = dist / mode
    Cd = C(d, dmax)
    C3000 = C(3000*km, dmax)

    muf = (1 + (Cd / C3000) * (info['B'] - 1)) * info['fof2'] + info['gyf'] * (1 - d / dmax) / 2

    return muf

def field_strength(iono, freq, from_lat, from_lon, to_lat, to_lon, longpath=False):
    ## Geodesy setup
    one_from = np.ndim(from_lat) == 0 and np.ndim(from_lon) == 0
    to_lat, to_lon = np.atleast_1d(to_lat, to_lon)

    dist, bearing = geodesy.distance_bearing(from_lat, from_lon, to_lat, to_lon)
    reverse_bearing = (bearing + np.pi) % (2 * np.pi)

    if longpath:
        dist = 2 * np.pi - dist
        bearing, reverse_bearing = reverse_bearing, bearing


    midpoint_lat, midpoint_lon = geodesy.destination(from_lat, from_lon, bearing, 0.5 * dist)
    midpoint = calc_cp(iono, midpoint_lat, midpoint_lon)

    up_to_9000 = dist <= 9000 * km
    over_7000 = dist > 7000 * km

    ## E-layer control points
    up_to_4000 = dist <= 4000 * km
    e_n0 = np.floor(dist / emax) + 1
    e_d0 = dist / e_n0

    takeoff = elevation_angle(e_d0, 110)
    too_low = (takeoff < min_elev) & up_to_4000
    while too_low.any():
        print(np.sum(too_low), "too_low e_n0")
        e_n0[too_low] += 1
        e_d0 = dist / e_n0
        takeoff = elevation_angle(e_d0, 110)
        too_low = (takeoff < min_elev) & up_to_4000

    up_to_2000 = dist <= 2000 * km
    over_2000 = (dist > 2000 * km) & (dist <= 4000 * km)

    screen_foe = np.zeros_like(to_lat)
    # <= 2000km: E-screening CP @ midpoint
    if np.any(up_to_2000):
        screen_foe[up_to_2000] = midpoint['foe'][up_to_2000]

    # 2000km - 4000km: E-screening CP @ endpoints ± 1000km
    if np.any(over_2000):
        thousand_km = np.full_like(to_lat[over_2000], 1000*km)
        f_lat = from_lat if one_from else from_lat[over_2000]
        f_lon = from_lon if one_from else from_lon[over_2000]
        t_plus_1000_lat, t_plus_1000_lon = geodesy.destination(f_lat, f_lon, bearing[over_2000], thousand_km)
        r_minus_1000_lat, r_minus_1000_lon = geodesy.destination(to_lat[over_2000], to_lon[over_2000], reverse_bearing[over_2000], thousand_km)
        t_plus_1000_cp = calc_cp(iono, t_plus_1000_lat, t_plus_1000_lon)
        r_minus_1000_cp = calc_cp(iono, r_minus_1000_lat, r_minus_1000_lon)
        screen_foe[over_2000] = np.maximum(
            t_plus_1000_cp['foe'],
            r_minus_1000_cp['foe'],
        )


def muf_luf(iono, from_lat, from_lon, to_lat, to_lon, longpath=False):
    ## Geodesy setup
    one_from = np.ndim(from_lat) == 0 and np.ndim(from_lon) == 0
    to_lat, to_lon = np.atleast_1d(to_lat, to_lon)

    dist, bearing = geodesy.distance_bearing(from_lat, from_lon, to_lat, to_lon)
    reverse_bearing = (bearing + np.pi) % (2 * np.pi)

    if longpath:
        dist = 2 * np.pi - dist
        bearing, reverse_bearing = reverse_bearing, bearing

    midpoint_lat, midpoint_lon = geodesy.destination(from_lat, from_lon, bearing, 0.5 * dist)
    midpoint = calc_cp(iono, midpoint_lat, midpoint_lon)

    ## E-layer control points
    up_to_4000 = dist <= 4000 * km

    e_n0 = np.floor(dist / emax) + 1
    e_d0 = dist / e_n0

    takeoff = elevation_angle(e_d0, 110)
    too_low = (takeoff < min_elev) & up_to_4000
    while too_low.any():
        print(np.sum(too_low), "too_low e_n0")
        e_n0[too_low] += 1
        e_d0 = dist / e_n0
        takeoff = elevation_angle(e_d0, 110)
        too_low = (takeoff < min_elev) & up_to_4000

    up_to_2000 = dist <= 2000 * km
    over_2000 = (dist > 2000 * km) & (dist <= 4000 * km)

    ## E-layer basic MUF
    e_muf = np.zeros_like(to_lat)
    # <= 2000km: CP @ midpoint
    if np.any(up_to_2000):
        e_muf[up_to_2000] = nEMUF(
            iono,
            dist[up_to_2000],
            e_n0[up_to_2000],
            { k: v[up_to_2000] for k, v in midpoint.items() },
        )

    # 2000km - 4000km: CP @ endpoints ̄± 1000km
    if np.any(over_2000):
        thousand_km = np.full_like(to_lat[over_2000], 1000*km)
        f_lat = from_lat if one_from else from_lat[over_2000]
        f_lon = from_lon if one_from else from_lon[over_2000]
        t_plus_1000_lat, t_plus_1000_lon = geodesy.destination(f_lat, f_lon, bearing[over_2000], thousand_km)
        r_minus_1000_lat, r_minus_1000_lon = geodesy.destination(to_lat[over_2000], to_lon[over_2000], reverse_bearing[over_2000], thousand_km)
        e_muf[over_2000] = np.minimum(
            nEMUF(
                iono,
                dist[over_2000],
                e_n0[over_2000],
                calc_cp(iono, t_plus_1000_lat, t_plus_1000_lon),
            ),
            nEMUF(
                iono,
                dist[over_2000],
                e_n0[over_2000],
                calc_cp(iono, r_minus_1000_lat, r_minus_1000_lon),
            )
        )

    ## F2-layer control points

    dmax = np.clip(midpoint['dmax'], 0., 4000 * km)
    f2_n0 = np.floor(dist / dmax) + 1
    f2_d0 = dist / f2_n0

    takeoff = elevation_angle(f2_d0, midpoint['hr'])
    too_low = takeoff < min_elev
    while too_low.any():
        print(np.sum(too_low), "too_low F2_n0")
        f2_n0[too_low] += 1
        f2_d0[too_low] = dist[too_low] / f2_n0[too_low]
        takeoff[too_low] = elevation_angle(f2_d0[too_low], midpoint['hr'][too_low])
        too_low = takeoff < min_elev

    up_to_dmax = dist <= dmax
    over_dmax = dist > dmax
    dmax_to_9000 = over_dmax & (dist <= 9000 * km)
    over_7000 = dist > 7000 * km

    hoplen = 2 * np.abs(np.sin(f2_d0 / 2.) / np.cos(takeoff + f2_d0 / 2.))
    pathlen = f2_n0 * hoplen
    takeoff_true = takeoff

    ## F2-layer basic MUF
    f2_muf = np.zeros_like(to_lat)
    # <= dmax: CP @ midpoint
    if np.any(up_to_dmax):
        f2_muf[up_to_dmax] = nF2MUF(
            iono,
            dist[up_to_dmax],
            f2_n0[up_to_dmax],
            { k: v[up_to_dmax] for k, v in midpoint.items() },
        )

    # dmax - 9000km: CPs at endpoints ± d0 / 2
    muf_m1 = np.zeros_like(to_lat)
    if np.any(dmax_to_9000):
        f_lat = from_lat if one_from else from_lat[dmax_to_9000]
        f_lon = from_lon if one_from else from_lon[dmax_to_9000]
        t_plus_d0_lat, t_plus_d0_lon = geodesy.destination(f_lat, f_lon, bearing[dmax_to_9000], f2_d0[dmax_to_9000] / 2)
        r_minus_d0_lat, r_minus_d0_lon = geodesy.destination(to_lat[dmax_to_9000], to_lon[dmax_to_9000], reverse_bearing[dmax_to_9000], f2_d0[dmax_to_9000] / 2)

        muf_m1[dmax_to_9000] = np.minimum(
            nF2MUF(
                iono,
                dist[dmax_to_9000],
                f2_n0[dmax_to_9000],
                calc_cp(iono, t_plus_d0_lat, t_plus_d0_lon),
            ),
            nF2MUF(
                iono,
                dist[dmax_to_9000],
                f2_n0[dmax_to_9000],
                calc_cp(iono, r_minus_d0_lat, r_minus_d0_lon),
            )
        )

    ## Calculation of hop length for 300km reflection height and 4000km maxhop
    # Used by the over-7000km MUF, and the LUF.

    nm_300 = np.floor(dist / (4000 * km)) + 1
    dm_300 = dist / nm_300

    takeoff = elevation_angle(dm_300, 300)
    too_low = takeoff < min_elev
    while too_low.any():
        print(np.sum(too_low), "too_low dm_300")
        nm_300[too_low] += 1
        dm_300[too_low] = dist[too_low] / nm_300[too_low]
        takeoff[too_low] = elevation_angle(dm_300[too_low], 300)
        too_low = takeoff < min_elev

    print("nm_300:", np.min(nm_300), "-", np.max(nm_300))

    # over 7000km: CPs at endpoints ± dm / 2
    muf_m2 = np.zeros_like(to_lat)
    if np.any(over_7000):
        f_lat = from_lat if one_from else from_lat[over_7000]
        f_lon = from_lon if one_from else from_lon[over_7000]

        t_plus_dm_lat, t_plus_dm_lon = geodesy.destination(f_lat, f_lon, bearing[over_7000], dm_300[over_7000] / 2)
        r_minus_dm_lat, r_minus_dm_lon = geodesy.destination(to_lat[over_7000], to_lon[over_7000], reverse_bearing[over_7000], dm_300[over_7000] / 2)

        cp1 = calc_cp(iono, t_plus_dm_lat, t_plus_dm_lon)
        cp2 = calc_cp(iono, r_minus_dm_lat, r_minus_dm_lon)

        f4_1 = 1.1 * cp1['fof2'] * cp1['M']
        f4_2 = 1.1 * cp2['fof2'] * cp2['M']
        fz_1 = cp1['fof2'] + 0.5 * cp1['gyf']
        fz_2 = cp2['fof2'] + 0.5 * cp2['gyf']

        fd = Fd(dm_300[over_7000])

        muf_cp1 = fz_1 + (f4_1 - fz_1) * fd
        muf_cp2 = fz_2 + (f4_2 - fz_2) * fd

        muf_m2[over_7000] = np.minimum(muf_cp1, muf_cp2)

    # Combine dmax-to-9000km values with 7000km-to-infinity values
    dlerp = np.clip(1. - (dist - 7000*km) / (2000*km), 0., 1.) # 1 at 7000km or less, 0 at 9000km or more
    f2_muf[over_dmax] = dlerp[over_dmax] * muf_m1[over_dmax] + (1. - dlerp[over_dmax]) * muf_m2[over_dmax]

    ## LUF
    # using the method from sec. 5.3.2 for >7000km field strength, but applied at every distance, because
    # that's a lot less computationally expensive than doing the full field-strength and SNR calculations
    # at a variety of frequencies, as recommended in section 9 and P.373.

    i90 = incidence_angle(elevation_angle(dm_300, 90.), 90.)

    slantlen = np.abs(np.sin(dm_300 / 2.) / np.cos(takeoff + dm_300 / 2.)) * nm_300

    luf_hops = np.max(nm_300).astype(int)
    sum_cos_chi = np.zeros_like(to_lat)

    for i in range(luf_hops):
        # We're assuming a reflection height of 300km and an E-layer penetration point of 90km
        # so using straight rays, the first penetration point is at (90/300) of a half-hop, or
        # i + 0.15 hops, and the second is at i+1 - 0.15 = i + 0.85 hops

        idx = i < nm_300
        f_lat = from_lat if one_from else from_lat[idx]
        f_lon = from_lon if one_from else from_lon[idx]

        p1_lat, p1_lon = geodesy.destination(f_lat, f_lon, bearing[idx], (i + 0.15) * dm_300[idx])
        p2_lat, p2_lon = geodesy.destination(f_lat, f_lon, bearing[idx], (i + 0.85) * dm_300[idx])

        sum_cos_chi[idx] += np.sqrt(cos_chi(iono.ts, p1_lat, p1_lon))
        sum_cos_chi[idx] += np.sqrt(cos_chi(iono.ts, p2_lat, p2_lon))

    # TODO winter anomaly
    luf = (5.3 * np.sqrt((1 + 0.009 * iono.ssn) * sum_cos_chi / (np.cos(i90) * np.log(9370 / slantlen))) - midpoint['gyf'])
    luf = np.maximum(luf, np.sqrt(dist / (3000 * km)))

    return {
        'muf_e': e_muf,
        'muf_f2': f2_muf,
        'muf': np.maximum(e_muf, f2_muf),
        'luf': luf,
        'distance': dist * r0,
        'hoplen': hoplen * r0,
        'pathlen': pathlen * r0,
        'khop': f2_n0.astype(float),
        'bearing': bearing,
        'takeoff': np.degrees(takeoff_true),
    }
