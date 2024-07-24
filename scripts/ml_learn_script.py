# Importation des modules nécessaires de PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, DateType

# Initialisation de la session Spark
spark = SparkSession.builder \
    .appName("ML Data Preparation") \
    .getOrCreate()

# Définition du schéma des données
schema = StructType([
    StructField("transaction_id", IntegerType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("price", FloatType(), True),
    StructField("total_amount", FloatType(), True),
    StructField("transaction_date", DateType(), True)
])

# Lecture des données CSV en utilisant le schéma défini
data = spark.read.csv("../data/sales_synthetic_data.csv", header=True, schema=schema)

# Préparation des données
# Création d'un vecteur de caractéristiques avec les colonnes 'quantity' et 'price'
feature_cols = ["quantity", "price"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Affichage des données transformées
data.show()

# Division des données en ensembles d'entraînement et de test
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Modèle de régression linéaire
# Initialisation du modèle de régression linéaire
lr = LinearRegression(featuresCol="features", labelCol="total_amount", regParam=0.1)

# Entraînement du modèle
lr_model = lr.fit(train_data)

# Prédictions sur les données de test
predictions = lr_model.transform(test_data)

# Affichage des résultats de la régression
predictions.select("features", "total_amount", "prediction").show()

# Évaluation du modèle de régression avec l'erreur quadratique moyenne (RMSE)
evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Optimisation des hyperparamètres
# Création d'un grid de paramètres pour la régression linéaire
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Configuration du CrossValidator
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse"),
                          numFolds=3)

# Entraînement du modèle avec CrossValidator
cv_model = crossval.fit(train_data)

# Prédictions avec le modèle optimisé
predictions = cv_model.transform(test_data)

# Évaluation du modèle optimisé
rmse = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse").evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) after Hyperparameter Tuning: {rmse}")

# Modèle de classification
# Création d'une colonne binaire 'label' pour indiquer les transactions de haute valeur
data = data.withColumn("label", when(col("total_amount") > 50, 1.0).otherwise(0.0))

# Recréation des ensembles d'entraînement et de test pour la classification
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Initialisation du modèle de régression logistique
lr_classifier = LogisticRegression(featuresCol="features", labelCol="label")

# Entraînement du modèle de classification
lr_model = lr_classifier.fit(train_data)

# Prédictions sur les données de test
predictions = lr_model.transform(test_data)

# Affichage des résultats de la classification
predictions.select("features", "label", "prediction").show()

# Évaluation du modèle de classification avec l'aire sous la courbe ROC (AUC)
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"Area Under ROC Curve (AUC): {auc}")

# Modèle de classification avancé
# Initialisation d'un arbre de décision pour la classification
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")

# Entraînement du modèle d'arbre de décision
dt_model = dt.fit(train_data)

# Prédictions avec le modèle d'arbre de décision
predictions = dt_model.transform(test_data)
predictions.show()

# Évaluation du modèle d'arbre de décision avec l'aire sous la courbe ROC (AUC)
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"Area Under ROC Curve (AUC) for Decision Tree: {auc}")
