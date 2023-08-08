# Databricks notebook source
# MAGIC %md
# MAGIC # Presentation

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- You can skip this line if no-UC
# MAGIC USE CATALOG mas;
# MAGIC
# MAGIC -- Sets the standard database to be used in this notebook
# MAGIC USE SCHEMA review_summarisation;

# COMMAND ----------

# 4x total core count
spark.conf.set("spark.sql.shuffle.partitions", 512)

# Read the table
reviews_df = spark.read.table("book_reviews_condensed")

# COMMAND ----------

from pyspark.sql import functions as SF
import html
meta_reviews_df = (
    reviews_df
    .withColumn("weighted_star_rating", SF.col("n_reviews") * SF.col("avg_star_rating"))
    .groupBy("asin", "title", "author", "week_start")
    .agg(
        SF.sum("n_reviews").alias("n_reviews"),
        SF.sum("n_review_tokens").alias("n_review_tokens"),
        SF.sum("weighted_star_rating").alias("weighted_star_rating")
    )
    .withColumn("avg_star_rating", SF.round(SF.col("weighted_star_rating") / SF.col("n_reviews"), 2))
    .drop("weighted_star_rating")
    .orderBy("asin", "title", "author", "week_start")
)

summary_reviews_df = (
    reviews_df
    .groupBy("asin", "title", "author", "week_start")
    .pivot("star_rating_class")
    .agg(SF.first("final_review_summary"))
    .withColumnRenamed("high", "positive_reviews_summary")
    .withColumnRenamed("low", "negative_reviews_summary")
    .orderBy("asin", "title", "author", "week_start")
)

summary_df = meta_reviews_df.join(summary_reviews_df, how="inner", on=["asin", "title", "author", "week_start"])

@SF.udf("string")
def convert_to_html(text):
    html_content = ""
    try:
        # Escape any existing HTML characters
        escaped_string = html.escape(text)
        # Replace newline characters with HTML line breaks
        html_content = escaped_string.replace('\n', '<br>')
    except:
        pass
    return html_content


summary_df = (
    summary_df
    .withColumn("positive_reviews_summary", convert_to_html("positive_reviews_summary"))
    .withColumn("negative_reviews_summary", convert_to_html("negative_reviews_summary"))
)
display(summary_df)

# COMMAND ----------

(
    summary_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("book_reviews_presentation")
)

# COMMAND ----------


