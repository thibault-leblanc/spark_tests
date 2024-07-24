from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType

def main():
    spark = SparkSession.builder \
        .appName("Dataset Example") \
        .getOrCreate()

    # Schéma explicite pour les données
    schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("price", FloatType(), True),
        StructField("total_amount", FloatType(), True),
        StructField("transaction_date", DateType(), True)
    ])

    # Lire le fichier CSV avec le schéma spécifié
    data = spark.read.csv("../data/sales_sample_data.csv", header=True, schema=schema)
    data.show()

    spark.stop()

if __name__ == "__main__":
    main()
