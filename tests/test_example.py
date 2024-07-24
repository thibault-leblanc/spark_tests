import unittest
from pyspark.sql import SparkSession

class ExampleTestCase(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSession.builder \
            .appName("Test") \
            .master("local[*]") \
            .getOrCreate()

    def tearDown(self):
        self.spark.stop()

    def test_example(self):
        data = [("Alice", 34), ("Bob", 45)]
        df = self.spark.createDataFrame(data, ["Name", "Age"])
        self.assertEqual(df.count(), 2)

if __name__ == '__main__':
    unittest.main()
