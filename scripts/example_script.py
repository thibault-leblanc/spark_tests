from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder \
        .appName("Example Spark Application") \
        .getOrCreate()

    # Exemple de lecture de données
    data = spark.read.csv("data/sample_data.csv", header=True, inferSchema=True)
    data.show()

    spark.stop()

if __name__ == "__main__":
    main()
