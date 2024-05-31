import wmm2020 as wmm
import jax.numpy as jnp

def sph_to_xyz(lat, lon):
    lon = lon * jnp.pi / 180.
    lat = lat * jnp.pi / 180.
    x = jnp.cos(lat) * jnp.cos(lon)
    y = jnp.cos(lat) * jnp.sin(lon)
    z = jnp.sin(lat)

    return x, y, z

def gen_coords(lat, lon, tm, dt):
    uthour = (tm % 86400) / 3600
    year_start = dt.replace(month=1, day=1)
    year_end = year_start.replace(year=year_start.year + 1)
    year_dec = dt.year + ((dt - year_start).total_seconds()) / float((year_end - year_start).total_seconds())

    mag = wmm.wmm(lat, lon, 300, year_dec)
    modip = jnp.degrees(jnp.arctan2(jnp.radians(mag.incl.item()), jnp.sqrt(jnp.cos(jnp.radians(lat)))))

    xyz_em = sph_to_xyz(modip, lon)
    xyz_sm = sph_to_xyz(modip, lon + 15. * uthour)
    xyz_sf = sph_to_xyz(lat, lon + 15. * uthour)

    return [tm / 86400., *xyz_em, *xyz_sm, *xyz_sf ]
