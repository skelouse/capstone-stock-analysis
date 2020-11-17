import os
from dotenv import load_dotenv
import MySQLdb
import pandas as pd

load_dotenv()


class DataBase():
    ip = os.getenv('IP')
    user = os.getenv('USER')
    password = os.getenv('PASSWORD')

    def __init__(self, db='stock', port=1433, host='localhost'):
        self.conn = MySQLdb.connect(host=host, password=self.password,
                                    port=port, db=db, user=self.user)
        self.cur = self.conn.cursor()

    def pull_names_as_dataframes(self, database_names):
        self.frames = {}
        for name in database_names:
            self.frames[name] = {}
            self.cur.execute("SELECT * FROM %s" % name)
            db_query = self.cur.fetchall()
            self.cur.execute("Show COLUMNS FROM %s" % name)
            cols_query = [col[0] for col in self.cur.fetchall()]
            self.frames[name] = pd.DataFrame(db_query,
                                             columns=cols_query)


def test():
    import time
    start = time.process_time()
    names = ['analyst']
    db = DataBase(names)
    print("Time taken = ", time.process_time() - start)
    print(db.frames['analyst'].columns)
