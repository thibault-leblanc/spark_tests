from pyspark import SparkConf, SparkContext

def get_spark_context(app_name="MySparkApp"):
    conf = SparkConf().setAppName(app_name).setMaster("local[*]")
    sc = SparkContext(conf=conf)
    return sc

