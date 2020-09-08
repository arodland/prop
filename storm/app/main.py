import datetime
import scipy.io
import urllib.request
import io
import os
import shutil

def mldt(ml):
    dt = datetime.datetime(year=1,month=1,day=1) + datetime.timedelta(days=ml) - datetime.timedelta(days=366) 
    # round to nearest minute
    dt += datetime.timedelta(seconds=30)
    dt -= datetime.timedelta(seconds=dt.second, microseconds=dt.microsecond)
    return dt

def kp_to_ap(kp):
    idx = int((kp + 0.1) * 3)
    return [0,2,3,4,5,6,7,9,12,15,18,22,27,32,39,48,56,67,80,94,111,132,154,179,207,236,300,400][idx]

with open('/iri-index/predicted/apf107.tmp', 'w') as out:
    nlines = 0
    with open('/iri-index/chain/apf107.dat') as fh:
        for line in fh:
            print(line, file=out, end='')
            nlines += 1

    data = {}
    with urllib.request.urlopen("https://spaceweather.gfz-potsdam.de/fileadmin/michaeli/rbm_forecast_v2/latest/Forecast_E_1_MeV_PA_50_UTC_latest.mat") as res:
        content = res.read()
        bio = io.BytesIO(content)
        mat = scipy.io.loadmat(bio)

        for time, kp in zip(mat['times'][0], mat['Kp'][0]):
            dt = mldt(time)
            dt += datetime.timedelta(seconds=30)
            dt -= datetime.timedelta(seconds=dt.second, microseconds=dt.microsecond)

            data[dt] = kp_to_ap(kp)

    keys_sorted = sorted(data.keys())
    first = keys_sorted[0]
    last = keys_sorted[-1]
    first_day = datetime.date(year=first.year, month=first.month, day=first.day)
    last_day = datetime.date(year=last.year, month=last.month, day=last.day)

    day = datetime.date(1958,1,1) + datetime.timedelta(days=nlines)

    while day <= last_day:
        ap = [0,0,0,0,0,0,0,0]
        ap_day = 0

        if day >= first_day:
            hourly = []
            for hour in range(24):
                dt = datetime.datetime(year=day.year, month=day.month, day=day.day, hour=hour)
                if dt < first:
                    hourly.append(data[first])
                elif dt > last:
                    hourly.append(data[last])
                else:
                    hourly.append(data[dt])

            for idx in range(8):
                ap[idx] = (hourly[3*idx] + hourly[3*idx + 1] + hourly[3*idx + 2])/3

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
