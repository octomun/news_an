from stock_mysql_64 import stock_setting
import pandas as pd
conn = stock_setting.connect_db()
stock_setting.drop_table_pro(conn)
stock_setting.create_table_pro(conn)


