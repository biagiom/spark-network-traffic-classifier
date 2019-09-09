import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, \
                                      NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import regexp_replace

spark = SparkSession.builder.appName("Network Attacks Classifier KDD99").master("local").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

start = time.time()

features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
            "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
            "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
            "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

dataset = spark.read.csv("kddcup.data.corrected", inferSchema=True, header=False)
dataset = dataset.toDF(*features)
dataset = dataset.withColumn("label", regexp_replace("label", "\.", ""))

print("Dataset sizes: {row} samples, {cols} features".format(row=dataset.count(), cols=len(dataset.columns)))
# dataset.printSchema()
# dataset.select("label").groupBy("label").count().orderBy("count", ascending=False).show(23)

categorical_features = ["protocol_type", "service", "flag"]
indexers = [StringIndexer(inputCol=column, outputCol=column + "_num") for column in categorical_features]
indexers.append(StringIndexer(inputCol="label", outputCol="label_num"))
pipeline = Pipeline(stages=indexers)
dataset = pipeline.fit(dataset).transform(dataset)

exclude_list = categorical_features + ["label", "label_num"]
# print("Exclude list:")
# print(exclude_list)
numerical_cols = [col for col in dataset.columns if col not in exclude_list]
# print(numerical_cols)

df_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="features")
dataset = df_assembler.transform(dataset)
# dataset.printSchema()

dataset = dataset.select(["features","label_num"])
# dataset.printSchema()

train_set, test_set = dataset.randomSplit([0.75, 0.25], seed=2019)
print("Training set Count: " + str(train_set.count()))
print("Test set Count: " + str(test_set.count()))

# Logistic Regression model
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0.8, featuresCol="features",
                        labelCol="label_num", family="multinomial")

# Decision Tree model
dt = DecisionTreeClassifier(labelCol="label_num", featuresCol="features",  maxBins=70)

# Random Forest model
rf = RandomForestClassifier(labelCol="label_num", featuresCol="features", numTrees=20, maxBins=70)

# Naive Bayes Multinomial
nb = NaiveBayes(labelCol="label_num", featuresCol="features", smoothing=1.0, modelType="multinomial")

classifiers = {"Logistic Regression": lr, "Decision Tree": dt,
               "Random Forest": rf, "Naive Bayes Multinomial": nb}

metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]

print("\nModels Evaluation:")
print("{:-<30}".format(""))
for c in classifiers:
	print(c)
	# fit the model
	model = classifiers[c].fit(train_set)
	
	# make predictions
	predictions = model.transform(test_set)
	predictions.cache()
	
	# evaluate performance
	evaluator = MulticlassClassificationEvaluator(labelCol="label_num", predictionCol="prediction")
	
	for m in metrics:
		evaluator.setMetricName(m)
		metric = evaluator.evaluate(predictions)
		print("{name} = {value:.2f}".format(name=m, value=metric))
	
	print("{:-<30}".format(""))

stop = time.time()
print("\nRunning time for Spark job '{name}': {time:.2f} s"
      .format(name=spark.conf.get("spark.app.name"), time=(stop-start)))
