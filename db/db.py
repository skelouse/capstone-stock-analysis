import os
from dotenv import load_dotenv
import MySQLdb
import pandas as pd

load_dotenv()


class DataBase():
    ip = os.getenv('IP')
    user = os.getenv('USER')
    password = os.getenv('PASSWORD')

    def __init__(self, database_names, port=14333,
                 db='stock', host='localhost'):
        conn = MySQLdb.connect(host='localhost', password=self.password,
                               port=port, db=db, user=self.user)
        cur = conn.cursor()
        self.frames = {}
        for name in database_names:
            cur.execute("SELECT * FROM %s" % name)
            self.frames[name] = pd.DataFrame(cur.fetchall())


if __name__ == "__main__":
    import time
    start = time.process_time()
    names = ['analyst', 'analystranking', 'company', 'performance',
             'performancepercentiles', 'prices', 'splits']
    db = DataBase(names)
    print("Time taken = ", time.process_time() - start)
    print(len(db.frames['company']))

while True:
    pass