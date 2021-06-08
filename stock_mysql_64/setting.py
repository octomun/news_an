import pymysql
news_db = pymysql.connect(
    user='root',
    passwd='@@qazsx852',
    host='127.0.0.1',
    db='news_db',
    charset='utf8'
)
def create_table():
    try:
        cursor = news_db.cursor()
        sql="""
            CREATE TABLE url (date varchar(20) NOT NULL,
            href TEXT NOT NULL);"""
        cursor.execute(sql)
        news_db.commit()
    finally:
        news_db.close()

def insert_tabel( url, href):
    try:
        cursor = news_db.cursor()
        sql="""
            insert into url(date, href) values(%s, %s)"""
        cursor.execute(sql,(url, href))
        news_db.commit()
    finally:
        news_db.close()

def fetch_all():
    try:
        cursor = news_db.cursor()
        sql = """
            select * from url"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        news_db.close()

# def delete_table():
#     try:
#         cursor = news_db.cursor()
#         sql = """
#             delete from url"""
#         cursor.execute(sql)
#         news_db.commit()
#     finally:
#         news_db.close()
#
# def delete_whare():
#     try:
#         cursor = news_db.cursor()
#         sql = """
#             delete from url"""
#         cursor.execute(sql)
#         news_db.commit()
#     finally:
#         news_db.close()

def drop_table():
    try:
        cursor = news_db.cursor()
        sql = """
            drop table url"""
        cursor.execute(sql)
        news_db.commit()
    finally:
        news_db.close()