import pymysql
from argparse import Namespace
import pandas as pd

MariaDB = Namespace(
    hostname="database-1.c1pvc7savwst.ap-northeast-2.rds.amazonaws.com",
    username="nsuser",
    password="nsuser))))",
    database="NewsSalad_dev_01",
    charset='utf8',
    tb_name='tb_event_seibro'
)

MariaDB_pro = Namespace(
    hostname="newssalad-prod-mariadb.c1dqccjyolma.ap-northeast-2.rds.amazonaws.com",
    username="admin",
    password="admin!#123",
    database="NewsSalad_Pro_01",
    charset='utf8',
    tb_name='tb_event_seibro'
)


def set_mariadb(hostname_t, username_t, password_t, db_name_t, charset_t):
    return pymysql.connect(host=hostname_t, user=username_t, passwd=password_t, database=db_name_t, charset = charset_t)

maria = set_mariadb(MariaDB.hostname,
                    MariaDB.username,
                    MariaDB.password,
                    MariaDB.database,
                    MariaDB.charset)

maria_pr = set_mariadb(MariaDB_pro.hostname,
                    MariaDB_pro.username,
                    MariaDB_pro.password,
                    MariaDB_pro.database,
                    MariaDB_pro.charset)
curs = maria.cursor(pymysql.cursors.DictCursor)
curs_pr = maria_pr.cursor(pymysql.cursors.DictCursor)

query = '''
SELECT name_nasdaq, type 
FROM TB_STOCK_MASTER_US 
WHERE std_dt = '20210910'
AND exchange in ('NASDAQ','NYSE')'''

curs.execute(query)
data = curs.fetchall()
f = open('/home/james/data/PycharmProjects/BASIC-NLP-PRACTICE/RNN_stock_type/data/type_data.csv',mode = 'wt', encoding = 'utf-8')
for _ in data:
    if _["type"] is None:
        pass
    else:
        f.write(f'{_["type"]}|{_["name_nasdaq"]}\n')
f.close()