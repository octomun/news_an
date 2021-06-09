import mysql.connector
import pandas as pd

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

def fetch_all(db,db_name): #전부 출력
    try:
        cursor = db.cursor( buffered=True,dictionary=True)
        sql = f"""
            select * from {db_name};"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()

def drop_table(db,db_name):
    try:
        cursor = db.cursor(raw=True, buffered=True)
        sql = f"""
            drop table {db_name};"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()


def fetch_all_now(db, db_name):  # 전부 출력
    try:
        cursor = db.cursor(buffered=True, dictionary=True)
        sql = f"""
            select * from {db_name}_now;"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()


def drop_table_now(db, db_name):
    try:
        cursor = db.cursor(raw=True, buffered=True)
        sql = f"""
            drop table {db_name}_now;"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()
#--------------------------------------------------------------------------------------------
# def create_table_predict(db):
#     try:
#         cursor = db.cursor( buffered=True)
#         sql=f"""
#             CREATE TABLE stock_data_predict (
#             predict FLOAT(20) not null
#             );"""
#         cursor.execute(sql)
#         db.commit()
#     finally:
#         cursor.close()
#
# def insert_tabel_predict( predict,db):
#     try:
#         cursor = db.cursor( buffered=True)
#         sql=f"""
#             insert into stock_data_predict(predict)value({predict});"""
#         cursor.execute(sql)
#         db.commit()
#     finally:
#         cursor.close()
#
# def fetch_all_predict(db): #전부 출력
#     try:
#         cursor = db.cursor( buffered=True,dictionary=True)
#         sql = f"""
#             select * from stock_data_predict;"""
#         cursor.execute(sql)
#         result = cursor.fetchall()
#         return result
#     finally:
#         cursor.close()
#
# def drop_table_predict(db):
#     try:
#         cursor = db.cursor( buffered=True)
#         sql = f"""
#             drop table stock_data_predict;"""
#         cursor.execute(sql)
#         db.commit()
#     finally:
#         cursor.close()
#
# def delete_all_table_predict(db):
#     try:
#         cursor = db.cursor( buffered=True)
#         sql = f"""
#             DELETE FROM stock_data_predict;"""
#         cursor.execute(sql)
#         db.commit()
#     finally:
#         cursor.close()
#--------------------------------------------------------------------------------------
def create_table_predict_pro(db,db_name):
    try:
        cursor = db.cursor( buffered=True)
        sql=f"""
            CREATE TABLE predict_{db_name}_pro (
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

def insert_tabel_predict_pro( dates, times, opens, highs, lows, closes, vols,date_time,fdays_rolling,tdays_rolling, max_min, pct, week_day, month,db,db_name):
    try:
        # stock_db.reconnect()
        cursor = db.cursor( buffered=True)
        sql=f"""
            insert into predict_{db_name}_pro(dates, times, opens, highs, lows, closes, vols,date_time,5days_rolling, 10days_rolling, max_min, pct, week_day, month)
             values(%s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s,%s,%s,%s);"""
        cursor.execute(sql,(dates, times, opens, highs, lows, closes, vols,date_time,fdays_rolling, tdays_rolling, max_min, pct, week_day, month))
        db.commit()
    finally:
        cursor.close()



def fetch_all_predict_pro(db,db_name): #전부 출력
    try:
        cursor = db.cursor( buffered=True,dictionary=True)
        sql = f"""
            select * from predict_{db_name}_pro;"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()

def drop_table_predict_pro(db,db_name):
    try:
        cursor = db.cursor( buffered=True)
        sql = f"""
            drop table predict_{db_name}_pro;"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()

def delete_all_table_predict_pro(db,db_name):
    try:
        cursor = db.cursor( buffered=True)
        sql = f"""
            DELETE FROM predict_{db_name}_pro;"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()

#------------------------------------------------------------
def update_table_stock_code(db,code, predict):
    try:
        cursor = db.cursor(raw=True, buffered=True)
        sql=f"""
            UPDATE `stock`.`stock_data_code` SET `predict` = %s, `pct` = (%s-now/%s) WHERE (`code` = %s)"""
        cursor.execute(sql,(predict,predict,predict,code))
        db.commit()
    finally:
        cursor.close()


def fetch_all_stock_code(db): #전부 출력
    try:
        cursor = db.cursor(dictionary=True, buffered=True)
        sql = f"""
            select * from stock_data_code;"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()



#----------------------------------------------------------------
def create_table_model_pro(db,db_name):
    try:
        cursor = db.cursor( buffered=True)
        sql=f"""
            CREATE TABLE model_{db_name}_pro (
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

def insert_tabel_model_pro( dates, times, opens, highs, lows, closes, vols,date_time,fdays_rolling,tdays_rolling, max_min, pct, week_day, month,db,db_name):
    try:
        # stock_db.reconnect()
        cursor = db.cursor( buffered=True)
        sql=f"""
            insert into model_{db_name}_pro(dates, times, opens, highs, lows, closes, vols,date_time,5days_rolling, 10days_rolling, max_min, pct, week_day, month)
             values(%s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s,%s,%s,%s);"""
        cursor.execute(sql,(dates, times, opens, highs, lows, closes, vols,date_time,fdays_rolling, tdays_rolling, max_min, pct, week_day, month))
        db.commit()
    finally:
        cursor.close()



def fetch_all_model_pro(db,db_name): #전부 출력
    try:
        cursor = db.cursor( buffered=True,dictionary=True)
        sql = f"""
            select * from model_{db_name}_pro;"""
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    finally:
        cursor.close()

def drop_table_model_pro(db,db_name):
    try:
        cursor = db.cursor( buffered=True)
        sql = f"""
            drop table model_{db_name}_pro;"""
        cursor.execute(sql)
        db.commit()
    finally:
        cursor.close()