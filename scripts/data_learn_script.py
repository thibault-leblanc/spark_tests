from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType

def main():
    spark = SparkSession.builder \
        .appName("Dataset Example") \
        .getOrCreate()

    # Schéma explicite pour les données, permet de maîtriser le format des données
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

    #Lire le dataset
    data.show()

    #Lire 3 premières entrées du dataset
    data.show(3)

    # Afficher le schéma du DataFrame
    data.printSchema()

    #Filtrer les transactions dont le montant total est supérieur à 50
    high_value_transactions = data.filter(data.total_amount > 50)
    high_value_transactions.show()

    #Filtrer les transactions effectuées après une certaine date
    from pyspark.sql.functions import col
    filtered_transactions = data.filter(col("transaction_date") > "2023-01-20")
    filtered_transactions.show()

    #Sélectionner uniquement les colonnes transaction_id, customer_id et total_amount
    selected_columns = data.select("transaction_id", "customer_id", "total_amount")
    selected_columns.show()

    #Ajouter une colonne discounted_amount avec une remise de 10% sur le montant total
    data_with_discount = data.withColumn("discounted_amount", col("total_amount") * 0.90)
    data_with_discount.show()

    #Calculer le montant total des ventes par client
    from pyspark.sql.functions import sum
    sales_by_customer = data.groupBy("customer_id").agg(sum("total_amount").alias("total_sales"))
    sales_by_customer.show()

    #Calculer la quantité totale vendue par produit
    quantity_by_product = data.groupBy("product_id").agg(sum("quantity").alias("total_quantity"))
    quantity_by_product.show()

    #Calculer les statistiques de base (min, max, moyenne) sur le montant total
    from pyspark.sql.functions import min, max, avg
    stats = data.agg(
        min("total_amount").alias("min_amount"),
        max("total_amount").alias("max_amount"),
        avg("total_amount").alias("avg_amount")
    )
    stats.show()

    #Réaliser des jointures
    product_data = spark.createDataFrame([
        (5001, "Product A"),
        (5002, "Product B"),
        (5003, "Product C"),
        (5004, "Product D"),
        (5005, "Product E")
    ], ["product_id", "product_name"])

    joined_data = data.join(product_data, on="product_id", how="left")
    joined_data.show()

    #Trier les transactions par montant total en ordre décroissant
    sorted_data = data.orderBy(col("total_amount").desc())
    sorted_data.show()

    #Enregistrer le DataFrame transformé au format CSV
    data_with_discount.write.csv("../data/discounted_sales_data.csv", header=True)






    spark.stop()

if __name__ == "__main__":
    main()
