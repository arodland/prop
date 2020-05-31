import sys
import os
import time
import psycopg2

cwd = os.getcwd()

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
            .assign(station_id=row['id'],source='giro')
        df_list.append(df)
    stationdata=pd.concat(df_list)
    logger.info('{} read_csv complete {}'.format(dt.datetime.now(),s))
    stationdata = stationdata[['station_id', 'time', 'cs', 'fof2', 'mufd', 'foes', 'foe', 'hmf2', 'tec']]

    #logger.info('{} getting processed records for {} from {} to {}'.format(dt.datetime.now(),s, fromDate, toDate))
    processed = pd.read_sql("SELECT * FROM measurement WHERE station_id = '{}' ORDER BY time DESC LIMIT 100".format(ss), con)

    stationdata.cs = stationdata.cs.astype(str)
    stationdata = stationdata[stationdata.cs.str.contains("No") == False]

    #added processing 

    #get same datatypes before concat
    processed[['time']] = processed[['time']].apply(pd.to_datetime)
    stationdata[['time']] = stationdata[['time']].apply(pd.to_datetime)

    logger.info('{} row count: stationdata {} processed {}'.format(s, len(stationdata), len(processed)))
    concatted = pd.concat([processed,stationdata], ignore_index=True, sort=False).drop_duplicates(subset=['station_id', 'time'])
    concatted = concatted[pd.isnull(concatted['id'])]
    #logger.info ('row count: combined {}'.format(concatted.count()))

    #to shift 0 to 360 to -180 to 180 values for correct mapping, sun altitude
    stationdf.loc[stationdf.longitude > 180, 'longitude'] = stationdf.longitude - 360

    #merge to get station data
    stationdf.reset_index(inplace=True)
    concatted['station_id'] = concatted['station_id'].astype(int)
    unprocessed = stationdf.merge(concatted, left_on='id', right_on='station_id', how='right')

    #set --- to nan
    unprocessed = unprocessed.applymap(lambda x: None if type(x) is str and x == '---' else x)

    #filter out errors
    #unprocessed = unprocessed[unprocessed.time != 'ERROR:']

    #unprocessed[['mufd']] = unprocessed[['mufd']].apply(pd.to_numeric)
    unprocessed[['time']] = unprocessed[['time']].apply(pd.to_datetime)
    unprocessed.longitude = unprocessed.longitude.astype(float)
    unprocessed.latitude = unprocessed.latitude.astype(float)

    unprocessed.sort_values(by=['time'], inplace=True)

    unprocessed = unprocessed[['time','cs','fof2','fof1','mufd','foes','foe','hf2','he','hme','hmf2','hmf1','yf2','yf1','tec','scalef2','fbes', 'station_id', 'source']]

    #logger.info('{} to_sql start {}'.format(dt.datetime.now(),s))
    unprocessed.to_sql('measurement', con=engine, if_exists='append', index=False)
    #logger.info('{} to_sql complete {}'.format(dt.datetime.now(), s))
    #print ('complete {} {} new records'.format(s, len(unprocessed.index)))

#get_data(sys.argv[1])
