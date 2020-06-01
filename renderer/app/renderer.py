import os
import io
import pathlib
from datetime import datetime, timezone
import urllib.request
import subprocess

import h5py
import pandas as pd
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

def draw_map(out_path, dataset, metric, ts, format, dots):
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

    plt.write(out_path + '.svg')

    subprocess.run(['/usr/local/bin/svgo', '--multipass', out_path + '.svg'], check=True)
    plt.write(out_path + '.png')

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

    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('RENDERER_PORT')), threaded=False, processes=4)
