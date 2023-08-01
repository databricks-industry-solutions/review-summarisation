# Databricks notebook source
# MAGIC %md
# MAGIC # Data Download
# MAGIC We begin our project with doing the necessary data setup and downloading the dataset we need The online retail giant [Amazon's Product Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) are publicly available via an easily downloadable route. Each row in the dataset equates a review written by a user, and also has other data points such as star ratings which we will get to explore later.. 
# MAGIC
# MAGIC **Set Up**
# MAGIC
# MAGIC This notebook is run on 13.2 ML Runtime.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initial Setup
# MAGIC
# MAGIC Setting up the necessary data holding objects such as Catalogs, Databases or Volumes are a great way to start projects on Databricks. These help us organise our assets with ease.
# MAGIC
# MAGIC Given this, we will use the next few cells of code to create a Catalog, a Database (Schema) within that catalog which will hold our tables, and also a Volume which will hold our files.
# MAGIC
# MAGIC _If Unity Catalog is not yet enabled on your workspace, please follow the instructions for alternatives. It is not required for this project_

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Creating a Catalog (Optional, skip if no-UC)
# MAGIC CREATE CATALOG IF NOT EXISTS mas;
# MAGIC
# MAGIC -- Select the Catalog as Default for this Notebook
# MAGIC -- If you would like to use your own Catalog, you can replace the name
# MAGIC -- (Optional, skip if no-UC)
# MAGIC USE CATALOG mas;
# MAGIC
# MAGIC -- Create a Database
# MAGIC CREATE DATABASE IF NOT EXISTS review_summarisation;
# MAGIC
# MAGIC -- Select the Database as Default
# MAGIC USE SCHEMA review_summarisation;
# MAGIC
# MAGIC -- Create a Volume (Optional, skip if no-UC)
# MAGIC CREATE VOLUME IF NOT EXISTS data_store;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setting Up Paths
# MAGIC
# MAGIC We will now set up our paths, which we will use for downloading and storing the data. This code will give you the option to select a `dbfs` path or any other path you might want to use for storing the raw files.

# COMMAND ----------

# Import the OS system to declare a ENV variable
import os

# Setting up the storage path (please edit this if you would like to store the data somewhere else)
main_storage_path = "/Volumes/mas/review_summarisation/data_store"

# Declaring as an Environment Variable 
os.environ["MAIN_STORAGE_PATH"] = main_storage_path

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # Confirming the variable made it through
# MAGIC echo $MAIN_STORAGE_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC #### Downloading the Data
# MAGIC Now, we can download the data from the public registry.. There are many datasets which are available in this source. They are grouped by category such as Books or Cameras. For this use case, we will focus on the books dataset as we might see reviews about the books we have read before.
# MAGIC
# MAGIC These datasets are in the form of compressed JSON. Our first task is going to be to download and unzip the data in the main location we have predefined, and we are going to execute this within a shell script, using the `curl` utility for download.
# MAGIC
# MAGIC _This part might take about 12 minutes_

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # Create a new folder in storage
# MAGIC export AMAZON_REVIEWS_FOLDER=$MAIN_STORAGE_PATH/amazon_reviews
# MAGIC mkdir -p $AMAZON_REVIEWS_FOLDER
# MAGIC
# MAGIC # Create a temporary folder on local disk
# MAGIC export TMP_DATA_FOLDER=/local_disk0/tmp_data_save
# MAGIC mkdir -p $TMP_DATA_FOLDER
# MAGIC
# MAGIC # Move to temp folder
# MAGIC cd $TMP_DATA_FOLDER
# MAGIC
# MAGIC # Download the data
# MAGIC curl -sO https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Books.json.gz &
# MAGIC curl -sO https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz &
# MAGIC wait
# MAGIC echo Download Complete
# MAGIC
# MAGIC # Unzip 
# MAGIC gunzip Books.json.gz &
# MAGIC gunzip meta_Books.json.gz &
# MAGIC wait
# MAGIC echo Unzipping Complete
# MAGIC
# MAGIC # Copy to Target
# MAGIC cp Books.json $AMAZON_REVIEWS_FOLDER/books.json &
# MAGIC cp meta_Books.json $AMAZON_REVIEWS_FOLDER/meta_books.json &
# MAGIC wait
# MAGIC echo Copying Complete
# MAGIC
# MAGIC # Display whats there
# MAGIC du -ah $AMAZON_REVIEWS_FOLDER

# COMMAND ----------

# MAGIC %md
# MAGIC #### Quick View on Data
# MAGIC
# MAGIC At this point, we downloaded two datasets from the source:
# MAGIC - `meta_books.json` Contains data about the products (metadata) such as title, price, ID..
# MAGIC - `books.json` Contains the actual reviews on the products.
# MAGIC
# MAGIC
# MAGIC Lets take a quick look into how many rows we have in each dataset, and what the data looks like

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # Get a count of total reviews
# MAGIC echo -e "Reviews Count" 
# MAGIC wc -l < $MAIN_STORAGE_PATH/amazon_reviews/books.json
# MAGIC
# MAGIC # Get a count of products (metadata)
# MAGIC echo -e "\nMetadata Count"
# MAGIC wc -l < $MAIN_STORAGE_PATH/amazon_reviews/meta_books.json

# COMMAND ----------

# MAGIC %sh 
# MAGIC
# MAGIC # Preview Reviews
# MAGIC echo -e "Reviews Example"
# MAGIC head -n 1 $MAIN_STORAGE_PATH/amazon_reviews/books.json
# MAGIC
# MAGIC # Preview Metadata (Books)
# MAGIC echo -e "\nMetadata Example"
# MAGIC head -n 1 $MAIN_STORAGE_PATH/amazon_reviews/meta_books.json

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading as PySpark Dataframes
# MAGIC
# MAGIC Our data is in JSON format, and from the above example, we can see what the structure of the JSON looks like. We can move on the creating schemas for each datasets and then read them as PySpark Dataframes

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Reviews Table

# COMMAND ----------

# Imports
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    BooleanType,
    IntegerType,
    LongType,
)

# Define the reviews JSON schema
reviews_schema = StructType(
    [
        StructField("overall", FloatType(), True),
        StructField("verified", BooleanType(), True),
        StructField("reviewTime", StringType(), True),
        StructField("reviewerID", StringType(), True),
        StructField("asin", StringType(), True),
        StructField("reviewerName", StringType(), True),
        StructField("reviewText", StringType(), True),
        StructField("summary", StringType(), True),
        StructField("unixReviewTime", LongType(), True),
    ]
)

# Read the JSON file
raw_reviews_df = spark.read.json(
    f"{main_storage_path}/amazon_reviews/books.json",
    mode="DROPMALFORMED",
    schema=reviews_schema
)

# Repartition
raw_reviews_df = raw_reviews_df.repartition(128)

# Get count
print(f"Table row count: {raw_reviews_df.count()}")

# Display
display(raw_reviews_df.limit(2))


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Books Table

# COMMAND ----------

# Imports
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    BooleanType,
)

# Define the books JSON schema
books_schema_schema = StructType(
    [
        StructField("category", ArrayType(StringType()), True),
        StructField("tech1", StringType(), True),
        StructField("description", ArrayType(StringType()), True),
        StructField("fit", StringType(), True),
        StructField("title", StringType(), True),
        StructField("also_buy", ArrayType(StringType()), True),
        StructField("tech2", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("feature", ArrayType(StringType()), True),
        StructField("rank", StringType(), True),
        StructField("also_view", ArrayType(StringType()), True),
        StructField("main_cat", StringType(), True),
        StructField("similar_item", StringType(), True),
        StructField("date", StringType(), True),
        StructField("price", StringType(), True),
        StructField("asin", StringType(), True),
        StructField("imageURL", ArrayType(StringType()), True),
        StructField("imageURLHighRes", ArrayType(StringType()), True),
    ]
)

# Read the JSON file
raw_books_df = spark.read.json(
    f"{main_storage_path}/amazon_reviews/meta_books.json",
    mode="DROPMALFORMED",
    schema=books_schema_schema,
)

# Get row count
print(f"Table row count: {raw_books_df.count()}")

# Repartition
raw_books_df = raw_books_df.repartition(128)

# Display
display(raw_books_df.limit(2))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Joining Two Tables
# MAGIC By having a quick look above, we can tell that the data is in the format we expected it to be.. There are some columns which look redundant in the products (metadata) table, however we can deal with those in the next notebook where we will do pre-processing & exploration work.
# MAGIC
# MAGIC Whats also important is that the row counts of the dataframes are matching with the counts we got with our shell command, which means that we do not have any malformed records or data loss in the read process.
# MAGIC
# MAGIC Lets go ahead and join the two tables together to create a `book_reviews_df` which will have both metadata and reviews in a single place. We expect the row count of this one to be the same as the reviews row count if there are no mismatches..
# MAGIC
# MAGIC We can use the `asin` column to join, which is the id of the products, and execute an inner join

# COMMAND ----------

# Join and Create the new df
raw_book_reviews_df = raw_books_df.join(raw_reviews_df, how="inner", on=["asin"])

# Partition
raw_book_reviews_df = raw_book_reviews_df.repartition(128)

# Get a count
print(f"DF row count: {raw_book_reviews_df.count()}")

# Display the dataframe
display(raw_book_reviews_df.limit(2))

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like the number of rows have increased! This means we have some duplicates in the data which we will deal with in the next section.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save All Dataframes
# MAGIC
# MAGIC Final step is to save all of the dataframes we have as Delta tables, in the specific Schema we have created at the very top of this notebook.
# MAGIC
# MAGIC Even though we will probably only need the `raw_book_reviews` dataframe in the next sections, it is important to save the other two as well just in case we need to go back to them at some points.
# MAGIC
# MAGIC In the following section, we will specify some code to save. We do not need to specify the schema name since we have already done so at the very top of the notebook with the `USE SCHEMA` SQL command.
# MAGIC
# MAGIC We will also get to run an `OPTIMISE` command to ensure that the data is layed out in an optimal way in our lake.

# COMMAND ----------

# Save Raw Reviews
(
    raw_reviews_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("raw_reviews")
)

# Save Raw Books
(
    raw_books_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("raw_books")
)

# Save Book Reviews
(
    raw_book_reviews_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("raw_book_reviews")
)

# Optimize All
_ = spark.sql("OPTIMIZE raw_reviews;")
_ = spark.sql("OPTIMIZE raw_books;")
_ = spark.sql("OPTIMIZE raw_book_reviews;")
