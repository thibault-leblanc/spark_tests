from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from pyspark.sql.functions import col
import random
from datetime import datetime, timedelta
import pandas

# Initialiser SparkSession
spark = SparkSession.builder \
    .appName("Generate Synthetic Data") \
    .getOrCreate()

# Fonction pour générer une date aléatoire
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# Paramètres
num_records = 100
start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2023-12-31', '%Y-%m-%d')

# Fonction pour générer des données
def generate_data(num_records):
    data = []
    for _ in range(num_records):
        transaction_id = _ + 1
        customer_id = random.randint(1001, 1100)
        product_id = random.randint(5001, 5100)
        quantity = random.randint(1, 10)
        price = round(random.uniform(5.0, 50.0), 2)
        total_amount = round(quantity * price, 2)
        transaction_date = random_date(start_date, end_date).strftime('%Y-%m-%d')  # Format de date en chaîne
        data.append((transaction_id, customer_id, product_id, quantity, price, total_amount, transaction_date))
    return data

# Créer le DataFrame
data = generate_data(num_records)
schema = StructType([
    StructField("transaction_id", IntegerType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("price", FloatType(), True),
    StructField("total_amount", FloatType(), True),
    StructField("transaction_date", StringType(), True)  # Utiliser StringType pour la date
])
df = spark.createDataFrame(data, schema)

# Convertir la colonne de date en type DateType
df = df.withColumn("transaction_date", col("transaction_date").cast(DateType()))

# Sauvegarder en CSV
out_name="sales_synthetic_data"
# Convertir le DataFrame PySpark en Pandas DataFrame
pandas_df = df.toPandas()
# Sauvegarder en CSV avec pandas
pandas_df.to_csv('../data/'+out_name+'.csv', index=False)


# Afficher quelques lignes
df.show()
