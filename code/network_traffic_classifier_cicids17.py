import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, \
                                      NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import regexp_replace, col

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

spark = SparkSession.builder.appName("Network Attacks Classifier CICIDS2017").master("local").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

start = time.time()

dataset = spark.read.csv("*.pcap_ISCX.csv", inferSchema=True, header=True, multiLine=True,
                         ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)

print("Dataset schema:")
dataset.printSchema()

print("Original dataset sizes: {row} samples, {cols} features".format(row=dataset.count(), cols=len(dataset.columns)))
# replace 'REPLACEMENT CHARACTER' Unicode Character (�) u"\uFFFD" in some Label values
dataset = dataset.withColumn("Label", regexp_replace("Label", u"\uFFFD ", ""))
dataset.select("Label").groupBy("Label").count().orderBy("count", ascending=False).show()

dataset = dataset.where(col('Flow Duration') > 0) \
                 .where(col('Init_Win_bytes_forward') > 0) \
                 .where(col('Init_Win_bytes_backward') > 0) \
                 .where(col('Flow IAT Min') > 0) \
                 .where(col('Fwd IAT Min') > 0) \
                 .where(col('Fwd IAT Max') > 0)

print("\nParsed dataset sizes: {row} samples, {cols} features".format(row=dataset.count(), cols=len(dataset.columns)))
dataset.select("Label").groupBy("Label").count().orderBy("count", ascending=False).show()

features = [f for f in dataset.columns if f not in ["Label"]]
df_assembler = VectorAssembler(inputCols=features, outputCol="features").setHandleInvalid("skip")
dataset = df_assembler.transform(dataset)
# dataset.printSchema()

label_indexer = StringIndexer(inputCol="Label", outputCol="Label_Idx").setHandleInvalid("skip").fit(dataset)
dataset = label_indexer.transform(dataset)
# dataset.printSchema()

# dataset.select(["Label", "Label_Idx"]).distinct().orderBy("Label_Idx").show()
label_list = dataset.select(["Label", "Label_Idx"]).distinct().orderBy("Label_Idx").select("Label").rdd.flatMap(lambda x: x).collect()
# print(label_list)

dataset = dataset.select(["features","Label_Idx"])
# dataset.printSchema()

train_set, test_set = dataset.randomSplit([0.75, 0.25], seed=2019)
print("Training set Count: " + str(train_set.count()))
print("Test set Count: " + str(test_set.count()))

# Logistic Regression model
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0.8, featuresCol="features",
                        labelCol="Label_Idx", family="multinomial")

# Decision Tree model
dt = DecisionTreeClassifier(labelCol="Label_Idx", featuresCol="features", maxBins=len(features))

# Random Forest model
rf = RandomForestClassifier(labelCol="Label_Idx", featuresCol="features", numTrees=20, maxBins=len(features))

# Naive Bayes Multinomial
nb = NaiveBayes(labelCol="Label_Idx", featuresCol="features", smoothing=1.0, modelType="multinomial")

classifiers = {"Logistic Regression": lr, "Decision Tree": dt,
               "Random Forest": rf, "Naive Bayes Multinomial": nb}

metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]

print("\nModels Evaluation:")
print("{:-<24}".format(""))
for idx, c in enumerate(classifiers):
	print(c)
	# fit the model
	model = classifiers[c].fit(train_set)
	
	# make predictions
	predictions = model.transform(test_set)
	predictions.cache()
	
	# evaluate performance
	evaluator = MulticlassClassificationEvaluator(labelCol="Label_Idx", predictionCol="prediction")
	
	for m in metrics:
		evaluator.setMetricName(m)
		metric = evaluator.evaluate(predictions)
		print("{name} = {value:.2f}".format(name=m, value=metric))
	
	# Build confusion matrix using Scikit-learn (sktlearn)
	target_list = predictions.select("Label_Idx").rdd.flatMap(lambda x: x).collect()
	pred_list = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
	label_num_list = predictions.select("Label_Idx").distinct().orderBy("Label_Idx").rdd.flatMap(lambda x: x).collect()
	# print("\nClassification report using Sklearn:")
	# print(classification_report(target_list, pred_list, target_names=label_list))
	conf_matrix = confusion_matrix(target_list, pred_list, label_num_list)
	plt.figure(idx)
	plt.title("Confusion matrix - {model}".format(model=c))
	sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False,
	            annot_kws={"size": 7.5}, xticklabels=label_list, yticklabels=label_list)
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.draw()
	plt.tight_layout()
	
	print("{:-<24}".format(""))

stop = time.time()
print("\nRunning time for Spark job '{name}': {time:.2f} s"
      .format(name=spark.conf.get("spark.app.name"), time=(stop-start)))

plt.show()
