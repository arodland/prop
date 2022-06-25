import os
import io
import numpy as np
import raytrace
import raytrace.rec533
from iono import Iono
import h5py
from flask import Flask, request, make_response, jsonify

app = Flask(__name__)

@app.route('/moflof.h5', methods=['GET'])
def moflof():
    from_lat = np.array(float(request.args.get('lat', 41.024)))
    from_lon = np.array(float(request.args.get('lon', -74.138)))
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)
    res = float(request.args.get('res', '2'))

    from_lat *= np.pi / 180
    from_lon *= np.pi / 180

    iono_url = 'http://localhost:%s/assimilated.h5' % (os.getenv('API_PORT'))
    if run_id is not None and ts is not None:
        iono_url += '?run_id=%s&ts=%s' % (run_id, ts)

    iono = Iono(iono_url)

    to_lat = np.linspace(-90, 90, int(180 / res) + 1) * np.pi / 180
    to_lon = np.linspace(-180, 180, int(360 / res) + 1) * np.pi / 180

    to_lon, to_lat = np.meshgrid(to_lon, to_lat)

    rt_sp = raytrace.mof_lof(iono, from_lat, from_lon, to_lat, to_lon)
    mof_sp, lof_sp = rt_sp['mof'], rt_sp['lof']

    rt_lp = raytrace.mof_lof(iono, from_lat, from_lon, to_lat, to_lon, longpath=True)
    mof_lp, lof_lp = rt_lp['mof'], rt_lp['lof']

    bio = io.BytesIO()
    h5 = h5py.File(bio, 'w')

    h5.create_dataset('/essn/ssn', data=iono.h5['/essn/ssn'])
    h5.create_dataset('/essn/sfi', data=iono.h5['/essn/sfi'])
    h5.create_dataset('/ts', data=iono.h5['/ts'])
    h5.create_dataset('/maps/mof_sp', data=mof_sp, compression='gzip', scaleoffset=3, shuffle=True)
    h5.create_dataset('/maps/mof_lp', data=mof_lp, compression='gzip', scaleoffset=3, shuffle=True)
    h5.create_dataset('/maps/lof_sp', data=lof_sp, compression='gzip', scaleoffset=3, shuffle=True)
    h5.create_dataset('/maps/lof_lp', data=lof_lp, compression='gzip', scaleoffset=3, shuffle=True)

    h5.close()

    resp = make_response(bio.getvalue())
    resp.mimetype = 'application/x-hdf5'
    return resp

@app.route('/rec533.h5', methods=['GET'])
def rec533():
    from_lat = np.array(float(request.args.get('lat', 41.024)))
    from_lon = np.array(float(request.args.get('lon', -74.138)))
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)
    res = float(request.args.get('res', '2'))

    from_lat *= np.pi / 180
    from_lon *= np.pi / 180

    iono_url = 'http://localhost:%s/assimilated.h5' % (os.getenv('API_PORT'))
    if run_id is not None and ts is not None:
        iono_url += '?run_id=%s&ts=%s' % (run_id, ts)

    iono = Iono(iono_url)

    to_lat = np.linspace(-90, 90, int(180 / res) + 1) * np.pi / 180
    to_lon = np.linspace(-180, 180, int(360 / res) + 1) * np.pi / 180

    to_lon, to_lat = np.meshgrid(to_lon, to_lat)

    rt_sp = raytrace.rec533.rec_533(iono, from_lat, from_lon, to_lat, to_lon)
    rt_lp = raytrace.rec533.rec_533(iono, from_lat, from_lon, to_lat, to_lon, longpath=True)

    muf_ratio = np.maximum(rt_sp['muf'] / rt_sp['luf'], 0.8 * rt_lp['muf'] / rt_lp['luf'])

    bio = io.BytesIO()
    h5 = h5py.File(bio, 'w')

    h5.create_dataset('/essn/ssn', data=iono.h5['/essn/ssn'])
    h5.create_dataset('/essn/sfi', data=iono.h5['/essn/sfi'])
    h5.create_dataset('/ts', data=iono.h5['/ts'])
    h5.create_dataset('/maps/muf_e_sp', data=rt_sp['muf_e'], compression='gzip', scaleoffset=3)
    h5.create_dataset('/maps/muf_e_lp', data=rt_lp['muf_e'], compression='gzip', scaleoffset=3)
    h5.create_dataset('/maps/muf_f2_sp', data=rt_sp['muf_f2'], compression='gzip', scaleoffset=3)
    h5.create_dataset('/maps/muf_f2_lp', data=rt_lp['muf_f2'], compression='gzip', scaleoffset=3)
    h5.create_dataset('/maps/muf_sp', data=rt_sp['muf'], compression='gzip', scaleoffset=3)
    h5.create_dataset('/maps/muf_lp', data=rt_lp['muf'], compression='gzip', scaleoffset=3)
    h5.create_dataset('/maps/luf_sp', data=rt_sp['luf'], compression='gzip', scaleoffset=3)
    h5.create_dataset('/maps/luf_lp', data=rt_lp['luf'], compression='gzip', scaleoffset=3)
    h5.create_dataset('/maps/ratio', data=muf_ratio, compression='gzip', scaleoffset=3)

    h5.close()

    resp = make_response(bio.getvalue())
    resp.mimetype = 'application/x-hdf5'
    return resp


@app.route('/ptp.json', methods=['GET'])
def ptp_json():
    from_lat = np.array(float(request.args.get('from_lat', None)))
    from_lon = np.array(float(request.args.get('from_lon', None)))
    to_lat = np.array(float(request.args.get('to_lat', None)))
    to_lon = np.array(float(request.args.get('to_lon', None)))
    run_id = request.args.get('run_id', None)
    ts_list = request.args.getlist('ts')
    path = request.args.get('path', 'both')
    debug = request.args.get('debug', None)
    try:
        debug = int(debug)
    except:
        pass

    from_lat *= np.pi / 180
    from_lon *= np.pi / 180
    to_lat *= np.pi / 180
    to_lon *= np.pi / 180

    ret = []
    for ts in ts_list:
        iono_url = 'http://localhost:%s/assimilated.h5?run_id=%s&ts=%s' % (os.getenv('API_PORT'), run_id, ts)
        iono = Iono(iono_url)

        out = {
            'ts': int(ts),
            'metrics': {},
        }

        if path in ('short', 'both'):
            rt_sp = raytrace.rec533.rec_533(iono, from_lat, from_lon, to_lat, to_lon)
            keys = rt_sp.keys() if debug else ['muf', 'luf']
            out['metrics'].update({ k+'_sp': rt_sp[k].tolist() for k in keys })

        if path in ('long', 'both'):
            rt_lp = raytrace.mof_lof(iono, from_lat, from_lon, to_lat, to_lon, longpath=True)
            keys = rt_lp.keys() if debug else ['muf', 'luf']
            out['metrics'].update({ k+'_lp': rt_lp[k].tolist() for k in keys })

        if path == 'both' and debug:
            muf_ratio = np.maximum(rt_sp['muf'], rt_lp['muf']) / np.minimum(rt_sp['luf'], rt_lp['luf'])
            out['metrics']['muf_ratio'] = muf_ratio.tolist()

        ret.append(out)

    return jsonify(ret)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('RAYTRACE_PORT')), threaded=False, processes=8)
