import datetime
import scipy.io
import urllib.request
import io
import os
import shutil

def kp_to_ap(kp):
    idx = int((kp + 0.1) * 3)
    return [0,2,3,4,5,6,7,9,12,15,18,22,27,32,39,48,56,67,80,94,111,132,154,179,207,236,300,400][idx]

with open('/iri-index/predicted/apf107.tmp', 'w') as out:
    nlines = 0
    with open('/iri-index/chain/apf107.dat') as fh:
        for line in fh:
            print(line, file=out, end='')
            nlines += 1

    forecast = {}
    ql = {}

    with urllib.request.urlopen("https://spaceweather.gfz-potsdam.de/fileadmin/ruggero/Kp_forecast/forecast_figures/KP_FORECAST_CURRENT.mat") as res:
        content = res.read()
        bio = io.BytesIO(content)
        mat = scipy.io.loadmat(bio)

        for time, kp in zip(mat['t'], mat['kp'][0]):
            dt = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            forecast[dt] = kp_to_ap(kp)

    with urllib.request.urlopen("http://www-app3.gfz-potsdam.de/kp_index/Kp_ap_nowcast.txt") as res:
        # alt: ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107/Kp_ap_nowcast.txt
        for line in res:
            line = line.decode('utf-8')
            if line.startswith("#"):
                continue
            fields = line.split()
            dt = datetime.datetime(
                year = int(fields[0]),
                month = int(fields[1]),
                day = int(fields[2]),
                hour = int(float(fields[3])),
                minute = 0,
                second = 0,
            )
            ap = int(fields[8])
            if ap == -1:
                continue

            ql[dt] = ap

    keys_sorted = sorted(forecast.keys())
    fc_first = keys_sorted[0]
    fc_last = keys_sorted[-1]
    fc_first_day = datetime.date(year=fc_first.year, month=fc_first.month, day=fc_first.day)
    fc_last_day = datetime.date(year=fc_last.year, month=fc_last.month, day=fc_last.day)

    keys_sorted = sorted(ql.keys())
    ql_first = keys_sorted[0]
    ql_last = keys_sorted[-1]
    ql_first_day = datetime.date(year=ql_first.year, month=ql_first.month, day=ql_first.day)
    ql_last_day = datetime.date(year=ql_last.year, month=ql_last.month, day=ql_last.day)

    day = datetime.date(1958,1,1) + datetime.timedelta(days=nlines)

    while day <= fc_last_day:
        ap = [0,0,0,0,0,0,0,0]
        ap_day = 0

        if day >= ql_first_day:
            for idx in range(8):
                dt = datetime.datetime(year=day.year, month=day.month, day=day.day, hour=idx*3)
                if dt < ql_first:
                    ap[idx] = ql[ql_first]
                elif dt <= ql_last:
                    ap[idx] = ql[dt]
                elif dt < fc_first:
                    ap[idx] = forecast[fc_first]
                elif dt <= fc_last:
                    ap[idx] = forecast[dt]
                else:
                    ap[idx] = forecast[fc_last]

            ap_day = round(sum(ap) / len(ap))
            ap = [round(a) for a in ap]

        print('{:3d}{:3d}{:3d}{:3d}{:3d}{:3d}{:3d}{:3d}{:3d}{:3d}{:3d}{:3d}{:3d}{:5.1f}{:5.1f}{:5.1f}'.format(
            day.year % 100,
            day.month,
            day.day,
            *ap,
            ap_day,
            -11,
            0,
            0,
            0,
            ), file=out
        )
        day += datetime.timedelta(days=1)

os.rename('/iri-index/predicted/apf107.tmp', '/iri-index/predicted/apf107.dat')
shutil.copy('/iri-index/chain/ig_rz.dat', '/iri-index/predicted/ig_rz.tmp')
os.rename('/iri-index/predicted/ig_rz.tmp', '/iri-index/predicted/ig_rz.dat')
