from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType

# Initialiser SparkSession
spark = SparkSession.builder \
    .appName("ML Data Preparation") \
    .getOrCreate()

# Lecture et formattage des données
schema = StructType([
    StructField("transaction_id", IntegerType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("price", FloatType(), True),
    StructField("total_amount", FloatType(), True),
    StructField("transaction_date", DateType(), True)
])

data = spark.read.csv("../data/sales_synthetic_data.csv", header=True, schema=schema)

## Préparation des données
feature_cols = ["quantity", "price"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

data.show()

# Diviser les données en ensemble d'entraînement et ensemble de test
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

train_data.show()
test_data.show()


###Simple modèle de régression linéaire

from pyspark.ml.regression import LinearRegression
##Préparation et entrainement du modèle
# Initialiser le modèle
lr = LinearRegression(featuresCol="features", labelCol="total_amount", regParam=0.1)

# Entraîner le modèle
lr_model = lr.fit(train_data)

# Faire des prédictions
predictions = lr_model.transform(test_data)

# Afficher les résultats
predictions.select("features", "total_amount", "prediction").show()

##Evalutaion du modèle
from pyspark.ml.evaluation import RegressionEvaluator

# Initialiser l'évaluateur
evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")

# Calculer la RMSE
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

