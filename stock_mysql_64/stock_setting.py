import mysql.connector

config = {
    'user':'root',
    'password':'@@qazsx852',
    'host':'127.0.0.1',
    'database':'stock',
    'charset':'utf8'
}
def connect_db():
    stock_db = mysql.connector.connect(**config)
    return stock_db

def fetch_all(db): #전부 출력
    try:
        cursor = db.cursor( buffered=True,dictionary=True)
        sql = f"""
            select * from stock_data;"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()
#--------------------------------------------------------------------------------------------
def create_table_predict(db):
    try:
        cursor = db.cursor( buffered=True)
        sql=f"""
            CREATE TABLE stock_data_predict (
            predict FLOAT(20) not null
            );"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()

def insert_tabel_predict( predict,db):
    try:
        cursor = db.cursor( buffered=True)
        sql=f"""
            insert into stock_data_predict(predict)value({predict});"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()

def fetch_all_predict(db): #전부 출력
    try:
        cursor = db.cursor( buffered=True,dictionary=True)
        sql = f"""
            select * from stock_data_predict;"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()

def drop_table_predict(db):
    try:
        cursor = db.cursor( buffered=True)
        sql = f"""
            drop table stock_data_predict;"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()

def delete_all_table_predict(db):
    try:
        cursor = db.cursor( buffered=True)
        sql = f"""
            DELETE FROM stock_data_predict;"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()
#--------------------------------------------------------------------------------------
def create_table_pro(db):
    try:
        cursor = db.cursor( buffered=True)
        sql=f"""
            CREATE TABLE stock_data_pro (
            dates DATE NOT NULL,
            times CHAR(10) NOT NULL,
            opens FLOAT(20) not null,
            highs FLOAT(20) not null,
            lows  FLOAT(20) not null,
            closes  FLOAT(20) not null,
            vols  FLOAT(20) not null,
            date_time Date not null,
            5days_rolling FLOAT(20) not null,
            10days_rolling FLOAT(20) not null,
            max_min FLOAT(20) not null,
            pct FLOAT(20) not null,
            week_day int(2) not null,
            month int(2) not null
            );"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()

def insert_tabel_pro( dates, times, opens, highs, lows, closes, vols,date_time,fdays_rolling,tdays_rolling, max_min, pct, week_day, mohth,db):
    try:
        # stock_db.reconnect()
        cursor = db.cursor( buffered=True)
        sql=f"""
            insert into stock_data_pro(dates, times, opens, highs, lows, closes, vols,date_time,5days_rolling, 10days_rolling, max_min, pct, week_day, mohth)
             values(%s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s,%s,%s,%s);"""
        cursor.execute(sql,(dates, times, opens, highs, lows, closes, vols,date_time,fdays_rolling, tdays_rolling, max_min, pct, week_day, mohth))
        db.commit()
    finally:
        cursor.close()



def fetch_all_pro(db): #전부 출력
    try:
        cursor = db.cursor( buffered=True,dictionary=True)
        sql = f"""
            select * from stock_data_pro;"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()

def drop_table_pro(db):
    try:
        cursor = db.cursor( buffered=True)
        sql = f"""
            drop table stock_data_pro;"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()

def delete_all_table_pro(db):
    try:
        cursor = db.cursor( buffered=True)
        sql = f"""
            DELETE FROM stock_data_pro;"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()

#------------------------------------------------------------

