# Databricks notebook source
# DBTITLE 1,Lending loan Dataset
# MAGIC 
# MAGIC %sql
# MAGIC 
# MAGIC select * from lc_loan_dataa

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC term - The number of payments on the loan. Values are in months and can be either 36 or 60
# MAGIC 
# MAGIC homeOwnership - The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
# MAGIC 
# MAGIC grade - LC assigned loan grade
# MAGIC 
# MAGIC purpose - A category provided by the borrower for the loan request.
# MAGIC 
# MAGIC intRate - Interest Rate on the loan
# MAGIC 
# MAGIC addrState - The state provided by the borrower in the loan application
# MAGIC 
# MAGIC loan_status - Current status of the loan
# MAGIC 
# MAGIC application_type - Indicates whether the loan is an individual application or a joint application with two co-borrowers
# MAGIC 
# MAGIC loan_amnt - The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
# MAGIC 
# MAGIC emp_length - Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
# MAGIC 
# MAGIC annual_inc - The self-reported annual income provided by the borrower during registration.
# MAGIC 
# MAGIC dti - A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
# MAGIC 
# MAGIC dti_joint - A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income
# MAGIC 
# MAGIC delinq_2yrs - The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
# MAGIC 
# MAGIC revol_util - Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
# MAGIC 
# MAGIC total_acc - The total number of credit lines currently in the borrower's credit file
# MAGIC 
# MAGIC num_tl_90g_dpd_24m - Number of accounts 90 or more days past due in last 24 months

# COMMAND ----------

# DBTITLE 1,converting table to spark dataframe
lc_df = spark.table('lc_loan_dataa')
display(lc_df)

# COMMAND ----------

from pyspark.sql.types import *
lc_df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, regexp_extract
from pyspark.sql.functions import col
lc_df=lc_df.withColumn("intrate_cleaned",regexp_replace(col("int_rate"), "%", ""))

# COMMAND ----------

lc_df.printSchema()

# COMMAND ----------

 display(lc_df)

# COMMAND ----------

columns_to_drop = ['int_rate', 'addr_state', 'emp_length','term','revol_util','dti','dti_joint']
lc_df = lc_df.drop(*columns_to_drop)

# COMMAND ----------

#Import all from `sql.types`
from pyspark.sql.types import *

# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 
# List of continuous features
CONTI_FEATURES  = ['intrate_cleaned']
# Convert the type
lc_df = convertColumn(lc_df, CONTI_FEATURES, DoubleType())
# Check the dataset
lc_df.printSchema()

# COMMAND ----------

#Import all from `sql.types`
from pyspark.sql.types import *

# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 
# List of continuous features
CONTI_FEATURES  = ['emplen_cleaned']
# Convert the type
lc_df = convertColumn(lc_df, CONTI_FEATURES, IntegerType())
# Check the dataset
lc_df.printSchema()

# COMMAND ----------

 display(lc_df)

# COMMAND ----------

# DBTITLE 1,Descriptive statistics

display(lc_df.describe())

# COMMAND ----------

lc_df.groupBy('bad_loan').count().show()

# COMMAND ----------

#***************************************************************
from pyspark.sql.functions import isnan, when, count, col
lc_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in lc_df.columns]).show()

# COMMAND ----------

lc_df.select('intrate_cleaned','loan_amnt','annual_inc').describe().show()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select grade, bad_loan, count(*) from lc_loan_dataa group by grade, bad_loan

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select cast(term_cleaned as int), bad_loan, count(bad_loan) from lc_loan_dataa group by term_cleaned, bad_loan order by cast(term_cleaned as int)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select cast(emplen_cleaned as int), bad_loan, count(bad_loan) from lc_loan_dataa group by emplen_cleaned, bad_loan order by cast(emplen_cleaned as int)

# COMMAND ----------

lc_df.stat.crosstab("grade", "application_type").show()

# COMMAND ----------

lc_df.stat.crosstab("term_cleaned", "application_type").show()

# COMMAND ----------

lc_df.stat.crosstab("bad_loan", "application_type").show()

# COMMAND ----------

lc_df.stat.freqItems(["home_ownership", "installment", "grade", "purpose", "loan_status", "application_type", "loan_amnt", "annual_inc", "delinq_2yrs","revol_bal", "total_acc", "num_tl_90g_dpd_24m", "intrate_cleaned", "emplen_cleaned"], 0.6).collect()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select purpose, bad_loan, count(*) from lc_loan_dataa group by purpose, bad_loan

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select home_ownership, bad_loan, count(*) from lc_loan_dataa group by home_ownership, bad_loan

# COMMAND ----------

display(lc_df)

# COMMAND ----------

# MAGIC %md training phase

# COMMAND ----------


badloan_df = lc_df
(train_data, test_data) = badloan_df.randomSplit([0.7,0.3], 24)


print("Records for training: " + str(train_data.count()))
print("Records for evaluation: " + str(test_data.count()))



# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder

#from distutils.version import LooseVersion


#from pyspark.ml import Pipeline
#from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

catColumns = ["term_cleaned", "home_ownership", "grade", "purpose", "loan_status", "application_type","num_tl_90g_dpd_24m"]



# COMMAND ----------

stages= []

for catCol in catColumns:

    stringIndexer = StringIndexer(inputCol=catCol, outputCol=catCol + "Index")

    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[catCol + "catVec"])

    stages += [stringIndexer, encoder]

# COMMAND ----------

stages


# COMMAND ----------

from pyspark.ml.feature import Imputer
imputer = Imputer(inputCols=["emplen_cleaned"], outputCols=["N_emplencleaned"])
stages += [imputer]

# COMMAND ----------

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="bad_loan", outputCol="label")
stages += [label_stringIdx]


# COMMAND ----------

temp=label_stringIdx.fit(train_data).transform(train_data)

# COMMAND ----------

temp.show(1)

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
numericCols = ["installment", "loan_amnt", "annual_inc", "delinq_2yrs","revol_bal", "revol_bal", "N_emplencleaned", "revolutil_cleaned", "total_acc", "intrate_cleaned","rev_avg","dti_cleaned"]
assembleInputs = assemblerInputs = [c + "catVec" for c in catColumns] + numericCols
assembler = VectorAssembler(inputCols=assembleInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

stages

# COMMAND ----------

pipeline = Pipeline().setStages(stages)
pipelineModel = pipeline.fit(train_data)



# COMMAND ----------

#display(pipeline.fit(train_data).transform(train_data))

# COMMAND ----------

trainprepDF = pipelineModel.transform(train_data)
testprepDF = pipelineModel.transform(test_data)

# COMMAND ----------

trainprepDF.head(1)
#testprepDF.head(1)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)


# Train model with Training Data
lrModel = lr.fit(trainprepDF)



# COMMAND ----------

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------

summary=lrModel.summary


# COMMAND ----------

accuracy = summary.accuracy
falsePositiveRate = summary.weightedFalsePositiveRate
truePositiveRate = summary.weightedTruePositiveRate
fMeasure = summary.weightedFMeasure()
precision = summary.weightedPrecision
recall = summary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s\nAreaUnderROC: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall, summary.areaUnderROC))

# COMMAND ----------

display(lrModel, trainprepDF, "ROC")

# COMMAND ----------

display(lrModel, trainprepDF, "fittedVsResiduals")