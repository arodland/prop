import sys
import os
import time
import psycopg2
from statsd import StatsClient

cwd = os.getcwd()

statsd = StatsClient(host=os.getenv('STATSD_HOST'))

@statsd.timer('prop.giro_loader.get_data')
def get_data(s, n=1):

    #import sys
    import ssl
    import datetime as dt
    import fileinput
    import numpy as np
    import logging
    from sqlalchemy.types import String, Integer
    from datetime import datetime
    from datetime import timedelta
    from logging.handlers import RotatingFileHandler
    import pandas as pd

    # lock to serialize console output
    import threading
    lock = threading.Lock()

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        #logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler(sys.stdout)
    ])

    logger = logging.getLogger()

    #logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

    logger.info('{} get_data.py start {}'.format(dt.datetime.now(), s))

    con = None

    try:
        dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
        con = psycopg2.connect(dsn)
    except:
        logger.error("I am unable to connect to the database")

    now = dt.datetime.now()

    from sqlalchemy import create_engine
    dsn = "postgresql://%s:%s@%s:5432/%s" % (os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_HOST"), os.getenv("DB_NAME"))
    engine = create_engine(dsn)

    #added 4/21 for SSL error workaround
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    fromDate = str((dt.datetime.now() - dt.timedelta(minutes=int(20))).strftime('%Y-%m-%dT%H:%M:%S'))
    toDate = str((dt.datetime.now() + dt.timedelta(minutes=int(5))).strftime('%Y-%m-%dT%H:%M:%S'))
    urlfrom = '&fromDate=' + fromDate
    urlto = '&toDate=' + toDate
    urldates = urlfrom + urlto

    #logger.info('{} getting data for {} from {} to {}'.format(dt.datetime.now(),s, fromDate, toDate)) 

    #get station_id given code
    stationdf = pd.read_sql_query("SELECT id, code, longitude, latitude FROM station WHERE code = '{}'".format(s), con)
    ss = pd.Series(stationdf['id'])
    ss = int(ss)
    #only get records from sql that we're getting from didbase to save resources
    since = datetime.now() - timedelta(days=n+1)
    nn = n+2

    #get data from GIRO, save to stationdata
    urlpt1 = "https://lgdc.uml.edu/common/DIDBGetValues?ursiCode="
    urlpt2 = "&charName=MUFD,hmF2,TEC,foF2,foE,foEs&DMUF=3000"
    df_list = []
    for index, row in stationdf.iterrows():
        logger.info('{} read_csv {} {}{}{}{}'.format(dt.datetime.now(), row['code'], urlpt1, row['code'], urlpt2, urldates))
        df=pd.read_csv(urlpt1 + row['code'] + urlpt2 + urldates,
            comment='#',
            delim_whitespace=True,
            parse_dates=[0],
            names = ['time', 'cs', 'fof2', 'qd1', 'mufd', 'qd2', 'foes', 'qd3', 'foe', 'qd4', 'hmf2', 'qd5', 'tec', 'qd6'])\
            .assign(station_id=int(row['id']))
        df_list.append(df)
    stationdata=pd.concat(df_list)
    logger.info('{} read_csv complete {}'.format(dt.datetime.now(),s))
    stationdata = stationdata[['station_id', 'time', 'cs', 'fof2', 'mufd', 'foes', 'foe', 'hmf2', 'tec']]

    stationdata.cs = stationdata.cs.astype(str)
    stationdata = stationdata[stationdata.cs.str.contains("No") == False]
    stationdata[['time']] = stationdata[['time']].apply(pd.to_datetime)

    #set --- to nan
    stationdata = stationdata.applymap(lambda x: None if type(x) is str and x == '---' else x)

    #filter out errors
    stationdata[['time']] = stationdata[['time']].apply(pd.to_datetime)

    stationdata.sort_values(by=['time'], inplace=True)

    stationdata = stationdata[['station_id', 'time','cs','fof2','mufd','foes','foe','hmf2','tec']]
    stationdata = stationdata.assign(source='giro')

    statsd.incr('prop.giro_loader.rows', len(stationdata))

    with engine.connect() as conn:
        with conn.begin():
            for n, row in stationdata.iterrows():
                conn.execute(
                    "insert into measurement (station_id, time, cs, fof2, mufd, foes, foe, hmf2, tec, source) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) on conflict do nothing",
                    tuple(row.array)
                )

#get_data(sys.argv[1])
