"""Runtime patch for PyIRI 0.1.7: memoise Apex_geo_qd.

PyIRI's SH entry points call Apex_geo_qd once per UT step on the same lat/lon arrays. The transform
depends only on the points and the year of `dtime` (coefficients are annual), so everything after
the first call is repeated work — and dominates run time (~85% in profiles). Cache by
(points, year, direction). Candidate for upstream; until then we pin pyiri==0.1.7.
"""
import functools

import numpy as np
import PyIRI.sh_library as sh

_orig = sh.Apex_geo_qd
_cache = {}
_MAX = 32


def _key(lat, lon, dtime, transform_type):
    lat = np.ascontiguousarray(lat, dtype=float)
    lon = np.ascontiguousarray(lon, dtype=float)
    return (lat.shape, lat.tobytes(), lon.tobytes(), int(np.mean(dtime.year)), transform_type)


@functools.wraps(_orig)
def Apex_geo_qd(Lat, Lon, dtime, transform_type):
    k = _key(Lat, Lon, dtime, transform_type)
    hit = _cache.get(k)
    if hit is None:
        hit = _orig(Lat, Lon, dtime, transform_type)
        if len(_cache) >= _MAX:
            _cache.pop(next(iter(_cache)))
        _cache[k] = hit
    return tuple(np.array(x, copy=True) for x in hit)


sh.Apex_geo_qd = Apex_geo_qd


# --- real_SH_func: one vectorised Legendre call for all (l, m) + cached normalisation table -------
import scipy.special as ss  # noqa: E402

_orig_sh = sh.real_SH_func


@functools.lru_cache(maxsize=None)
def _norm_table(lmax):
    L = np.arange(lmax + 1)[:, None]
    m = np.arange(lmax + 1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.sqrt((2 - (m == 0)) * (2 * L + 1) * ss.factorial(L - m) / ss.factorial(L + m))
    return np.where(m <= L, n, 0.0)  # (lmax+1, lmax+1), zero where m > l


@functools.wraps(_orig_sh)
def real_SH_func(theta, phi, lmax=29):
    import PyIRI.main_library as ml

    theta = ml.to_numpy_array(theta)
    phi = ml.to_numpy_array(phi)
    mlt_flag = theta.ndim == 1
    if mlt_flag:
        theta = theta[np.newaxis, :]
        phi = phi[np.newaxis, :]
    z = np.cos(theta)
    if phi.shape != z.shape:
        phi = np.broadcast_to(phi, z.shape)
    N_T, N_G = z.shape
    # P[l, m] depends on colatitude only. PyIRI passes the same grid at every UT step (the MLT frame
    # only changes phi), so compute the Legendre part once per point and broadcast over time.
    same_theta = N_T > 1 and all(np.array_equal(z[0], z[i]) for i in range(1, N_T))
    zz = z[:1] if same_theta else z
    P = ss.assoc_legendre_p_all(lmax, lmax, zz)[0][:, : lmax + 1]  # (lmax+1, lmax+1, n_t, N_G), orders -m..m in axis 1
    pole = zz == -1.0
    if pole.any():
        P = P.copy()
        for L in range(lmax + 1):
            P[L, 0][pole] = (-1.0) ** L
    P = P * _norm_table(lmax)[:, :, None, None]
    if same_theta:
        P = np.broadcast_to(P, (lmax + 1, lmax + 1, N_T, N_G))
    F_SH = np.empty(((lmax + 1) ** 2, N_T, N_G), dtype=float)
    ms = np.arange(1, lmax + 1)
    cos_m = np.cos(ms[:, None, None] * phi[None])
    sin_m = np.sin(ms[:, None, None] * phi[None])
    for L in range(lmax + 1):
        base = L * (L + 1)
        F_SH[base] = P[L, 0]
        if L:
            F_SH[base + 1 : base + L + 1] = P[L, 1 : L + 1] * cos_m[:L]
            F_SH[base - L : base][::-1] = P[L, 1 : L + 1] * sin_m[:L]
    return F_SH.squeeze(1) if mlt_flag else F_SH


sh.real_SH_func = real_SH_func
