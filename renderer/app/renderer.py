import os
import io
import pathlib
from datetime import datetime, timezone
import urllib.request
import subprocess

import h5py
import pandas as pd
import numpy as np
from flask import Flask, request, make_response

import plot

def get_dataset(url):
    with urllib.request.urlopen(url) as res:
        content = res.read()
        bio = io.BytesIO(content)
        h5 = h5py.File(bio, 'r')
        return h5

def filter_data(df, metric, ts):
    df = df.drop(df[df.time / 1000 < (ts - 3600)].index)
    df = df.dropna(subset=[metric])
    df = df.drop(df[df.cs < 0.249].index)

    return df

def draw_map(out_path, dataset, metric, ts, format, dots, file_formats):
    tm = datetime.fromtimestamp(ts, timezone.utc)

    plt = plot.Plot(metric, tm, decorations=(False if format=='bare' else True))
    if metric == 'mufd':
        plt.scale_mufd()
    elif metric == 'fof2':
        plt.scale_fof2()
    else:
        plt.scale_generic()

    zi = dataset['/maps/' + metric][:]
    plt.draw_contour(zi)

    dotjson, dot_df = None, None

    if dots == 'curr':
        dotjson = str(dataset['/stationdata/curr'][...])
    elif dots == 'pred':
        dotjson = str(dataset['/stationdata/pred'][...])

    if dotjson is not None:
        dot_df = pd.read_json(dotjson)
        dot_df = filter_data(dot_df, metric, ts)
        plt.draw_dots(dot_df, metric)

    if format != 'bare':
        plt.draw_title(metric, 'eSFI: %.1f, eSSN: %.1f' % (dataset['/essn/sfi'][...], dataset['/essn/ssn'][...]))

    if 'svg' in file_formats:
        plt.write(out_path + '.svg')
        subprocess.run(['/usr/local/bin/svgo', '--multipass', out_path + '.svg'], check=True)

    if 'png' in file_formats:
        plt.write(out_path + '.png')

    if 'jpg' in file_formats:
        plt.write(out_path + '.jpg')

    if 'station_json' in file_formats and dotjson is not None:
        with open(out_path + '_station.json', 'w') as f:
            dot_df.to_json(f, orient='records')

def mof_lof(dataset, metric, ts, lat, lon, centered, file_format):
    tm = datetime.fromtimestamp(ts, timezone.utc)
    plt = plot.Plot(metric, tm, decorations=True, centered=((lon, lat) if centered else None))
    maps = { name: dataset['/maps/' + name][:] for name in ('mof_sp', 'mof_lp', 'lof_sp', 'lof_lp') }

    if metric.startswith('mof_'):
        plt.scale_mufd('turbo')
    elif metric.startswith('lof_'):
        plt.scale_fof2('turbo')
    else:
        plt.scale_generic('turbo')

    if metric.endswith('_combined'):
        base_metric = metric[:len(metric)-9]
        sp = maps[base_metric + '_sp']
        sp_valid = maps['mof_sp'] > maps['lof_sp']

        lp = maps[base_metric + '_lp']
        lp_valid = maps['mof_lp'] > maps['lof_lp']

        if base_metric == 'mof':
            lp_valid = lp_valid & (lp > sp)
            sp_valid = sp_valid & ~lp_valid
        else:
            lp_valid = lp_valid & (lp < sp)
            sp_valid = sp_valid & ~lp_valid

        contour = sp
        contour[lp_valid] = lp[lp_valid]
        hatch = lp_valid
        blackout = ~(sp_valid | lp_valid)

    else:
        path = metric[len(metric)-3:]
        contour = maps[metric]
        hatch = None
        blackout = maps['lof' + path] >= maps['mof' + path]

    plt.draw_longpath_hatches(hatch)
    plt.draw_blackout(blackout)
    plt.draw_mofstyle(contour)
    plt.draw_dot(lon, lat, text='\u2605', color='red', alpha=0.6)

    plt.draw_title(metric, 'eSFI: %.1f, eSSN: %.1f' % (dataset['/essn/sfi'][...], dataset['/essn/ssn'][...]))
    bio = io.BytesIO()
    plt.write(bio, format=file_format)
    return bio.getvalue()

if __name__ == '__main__':
    app = Flask(__name__)

    @app.route('/rendersvg', methods=['POST'])
    def rendersvg():
        run_id = int(request.form['run_id'])
        ts = int(request.form['target'])
        metric = request.form['metric']
        name = request.form['name']
        format = request.form['format']
        dots = request.form['dots']
        file_formats = request.form.getlist('file_format')

        job_path = '/output/%d' % (run_id)
        pathlib.Path(job_path).mkdir(parents=True, exist_ok=True)
        out_path = '%s/%s-%s-%s' % (job_path, metric, format, name)

        h5 = get_dataset('http://localhost:%s/assimilated.h5?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

        draw_map(
            out_path = out_path,
            dataset = h5,
            metric = metric,
            ts = ts,
            format = format,
            file_formats = file_formats,
            dots = dots,
        )

        return make_response("OK\n")

    @app.route('/renderhtml', methods=['POST'])
    def renderhtml():
        run_id = int(request.form['run_id'])
        try:
            os.unlink('/output/current')
        except:
            pass

        os.symlink('%d' % (run_id), '/output/current')

        return make_response("OK\n")

    @app.route('/moflof.svg', methods=['GET'])
    def moflof():
        run_id = int(request.values['run_id'])
        ts = int(request.values['ts'])
        metric = request.values['metric']
        lat = float(request.values['lat'])
        lon = float(request.values['lon'])
        centered = request.values.get('centered') in ('true', '1')

        h5 = get_dataset('http://localhost:%s/moflof.h5?run_id=%d&ts=%d&lat=%f&lon=%f' % (os.getenv('RAYTRACE_PORT'), run_id, ts, lat, lon))

        svg = mof_lof(h5, metric, ts, lat, lon, centered, 'svg')
        resp = make_response(svg)
        resp.mimetype = 'image/svg+xml'
        return resp

    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('RENDERER_PORT')), threaded=False, processes=16)
