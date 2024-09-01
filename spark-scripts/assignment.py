import pyspark
import os
from dotenv import load_dotenv
from pathlib import Path
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql.window import Window
from pyspark.sql.functions import to_date, date_format, col, count, when,lag, sum, countDistinct



dotenv_path = Path('/resources/.env')
load_dotenv(dotenv_path=dotenv_path)


#error in airflow because unable to read the postgres_host and postgres_dw_db (read as null) 
#therefore those variable are being hard code
postgres_host = '172.27.130.106'
postgres_dw_db = 'warehouse'
postgres_user = os.getenv('POSTGRES_USER')
postgres_password = os.getenv('POSTGRES_PASSWORD')

#error in airflow while initiate and create spark context then spark session 
#set spark using sparksession.builder
spark = SparkSession.builder \
    .appName("Dibimbing") \
    .config("spark.jars", "/opt/postgresql-42.2.18.jar") \
    .config("spark.driver.extraClassPath", "/opt/postgresql-42.2.18.jar") \
    .config("spark.executor.extraClassPath", "/opt/postgresql-42.2.18.jar") \
    .getOrCreate()


#configure the jdbc_properties
jdbc_url = f'jdbc:postgresql://{postgres_host}/{postgres_dw_db}'
print(jdbc_url)
jdbc_properties = {
    'user': postgres_user,
    'password': postgres_password,
    'driver': 'org.postgresql.Driver',
    'stringtype': 'unspecified'
}


#read the table from postgresql 
retail_df = spark.read.jdbc(
    jdbc_url,
    'public.retail',
    properties=jdbc_properties
)

#using memory only 
retail_df = retail_df.persist(pyspark.StorageLevel.MEMORY_ONLY)


#change the data type
retail_df = retail_df \
              .withColumn("invoicedate", to_date("invoicedate", "yyyy-MM-dd")) \
              .withColumn("month_year", date_format("invoicedate", "yyyy-MM"))

retail_df = retail_df \
              .withColumn("invoiceno", col("invoiceno").cast(IntegerType())) 

retail_df.printSchema()

#cleaning the data
retail_df = retail_df.dropDuplicates(subset=['invoiceno'])

retail_df = retail_df.na.drop(subset=['invoiceno'])

retail_df = retail_df.na.drop(subset=['customerid'])

columns_to_capitalize = ['description', 'country'] 

for col_name in columns_to_capitalize:
    retail_df = retail_df.withColumn(col_name, F.initcap(F.col(col_name)))



#aggregations
retail_df = retail_df.withColumn('amount', F.round(F.col('quantity')*F.col('unitprice'),2))
#total sales per month 
total_sales_per_month_df = retail_df.groupBy("month_year").agg(F.round(F.sum("amount"), 2).alias("total_sales")).orderBy(col("month_year").asc())
#unique_customers_per_month
unique_customers_per_month_df = retail_df.groupBy("month_year").agg(countDistinct("customerid").alias("unique_customers")).orderBy(col("month_year").asc())
#total_transactions_per_month
total_transactions_per_month_df = retail_df.groupBy("month_year").agg(count("*").alias("total_transactions")).orderBy(col("month_year").asc())


#write the result to postgresql
total_sales_per_month_df.write.jdbc(
    url=jdbc_url,
    table="total_sales_per_month",  
    mode="overwrite",
    properties=jdbc_properties
)

unique_customers_per_month_df.write.jdbc(
    url=jdbc_url,
    table="unique_customers_per_month",  
    mode="overwrite",
    properties=jdbc_properties
)


total_transactions_per_month_df.write.jdbc(
    url=jdbc_url,
    table="total_transactions_per_month",  
    mode="overwrite",
    properties=jdbc_properties
)



#calculate retention rate and churn rate 
#select customerid and month_year
active_customers_df = retail_df.select("customerid", "month_year").distinct()

#create partition
window_data = Window.partitionBy("customerid").orderBy("month_year")

#create new column (previous_month) column
active_customers_df = active_customers_df.withColumn(
    "previous_month",
    lag("month_year", 1).over(window_data)
)

#calculate total customers 
total_customers = active_customers_df.select("customerid").distinct().count()

#calculate retained customers (those who have transactions in consecutive months)
retention_df = active_customers_df.filter(col("previous_month").isNotNull())
retained_customers = retention_df.select("customerid").distinct().count()

#calculate retention rate
retention_rate = retained_customers / total_customers if total_customers > 0 else 0
retention_rate = round(retention_rate, 2)

#calculate churn rate
churn_rate = 1 - retention_rate
churn_rate = round(churn_rate, 2)


#create df retention rate and churn rate
retention_churn_df = spark.createDataFrame(
    [(retention_rate, churn_rate)],
    ["retention_rate", "churn_rate"]
)

#write the result to postgresql
retention_churn_df.write.jdbc(
    url=jdbc_url,
    table="retention_churn",  
    mode="overwrite",
    properties=jdbc_properties
)

