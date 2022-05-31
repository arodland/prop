#!python3
import os
import threading
from queue import Queue
import time
import datetime as dt
import get_data_station
from get_data_station import get_data
#from myflaskapp import *
import sys
import logging
import pandas as pd
import psycopg2


cwd = os.getcwd()

threadcount = int(sys.argv[2])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        #logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()

logger.info('{} tread.py start'.format(dt.datetime.now()))

try:
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg2.connect(dsn)
except:
    logger.error("I am unable to connect to the database")

# lock to serialize console output
lock = threading.Lock()
maxage = None

def do_work(item):
    #time.sleep(1) # pretend to do some lengthy work.
    # Make sure the whole print completes or threads can mix up output in one line.
    logger.info ('start ' + item)
    logger.info('{} do_work item {}'.format(dt.datetime.now(), item))
    try:
        get_data(item, maxage)
    except Exception as e:
        logger.error(e)

    #with lock:
        #logging.info('{} do_work item {} {}'.format(dt.datetime.now(), threading.current_thread().name, item))
# The worker thread pulls an item from the queue and processes it
def worker():
    while True:
        item = q.get()
        logger.info('{} worker got item {} from queue'.format(dt.datetime.now(), item))
        do_work(item)
        q.task_done()
        logger.info('{} worker done w item {}'.format(dt.datetime.now(), item))

with con.cursor() as cur:
    cur.execute("select max(m.time) from measurement m where m.source='giro'")
    (lasttime,) = cur.fetchone()
    intervals = round((dt.datetime.now() - lasttime) / dt.timedelta(minutes = 15))
    maxage = intervals * 15 + 5

# Create the queue and thread pool.
q = Queue()
logger.info('{} ### creating queue with tread pool count {} ###'.format(dt.datetime.now(), threadcount))
for i in range(threadcount):
     t = threading.Thread(target=worker)
     t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
     t.start()

# stuff work items on the queue.
start = time.perf_counter()
stationdf = pd.read_sql("select code from station where giro=true or giro_optional=true and id not in (select distinct station_id from measurement where source='noaa' and time > now() - interval '3 hours') order by random()", con)
stationdf = pd.Series(stationdf['code'].values)
for item in stationdf:
    q.put(item)

q.join()       # block until all tasks are done
con.close()    # close db
logger.info('time: %.1f', (time.perf_counter() - start))
