import os
from dotenv import load_dotenv
import MySQLdb
import pandas as pd

load_dotenv()


class DataBase():
    ip = os.getenv('IP')
    user = os.getenv('USER')
    password = os.getenv('PASSWORD')

    def __init__(self, db='stock', port=1433, host='localhost', **kwargs):
        self.conn = MySQLdb.connect(host=host, password=self.password,
                                    port=port, db=db, user=self.user, **kwargs)
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
    names = ['performance']
    db = DataBase('stock', 14333, 'localhost')
    # print("Time taken = ", time.process_time() - start)
    # print(db.frames['peformance'].columns)
    db.pull_names_as_dataframes(names)
    df_performance = db.frames['performance']
    print(df_performance.loc[df_performance['TotalReturn1Yr'] == '-15.4\x10098'].iloc[0])

def test_push():
    db = DataBase('stocks_cleaned', 14333, 'localhost', 
                  charset='utf8', use_unicode=True)
    db.cur.executemany(
      """INSERT INTO prices (name, spam, eggs, sausage, price)
      VALUES (%s, %s, %s, %s, %s)""",
      [
      ("Spam and Sausage Lover's Plate", 5, 1, 8, 7.95 ),
      ("Not So Much Spam Plate", 3, 2, 0, 3.95 ),
      ("Don't Wany ANY SPAM! Plate", 0, 4, 3, 5.95 )
      ] )