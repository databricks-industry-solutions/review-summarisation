# Databricks notebook source
# MAGIC %md
# MAGIC # Condensation

# COMMAND ----------

# Install libraries
%pip install -qq flash-attn
%pip install -qq xformers
%pip install -qq torch==2.0.1
%pip install -qq ctranslate2==3.17
%pip install -qq triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python

# Restart Python Kernel
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- You can skip this line if no-UC
# MAGIC USE CATALOG mas;
# MAGIC
# MAGIC -- Sets the standard database to be used in this notebook
# MAGIC USE SCHEMA review_summarisation;
# MAGIC
# MAGIC -- Create a Volume (Optional, skip if no-UC)
# MAGIC CREATE VOLUME IF NOT EXISTS model_store;

# COMMAND ----------

# Import the OS system to declare a ENV variable
import os

# Setting up the storage path (please edit this if you would like to store the data somewhere else)
main_storage_path = "/Volumes/mas/review_summarisation/model_store"

# Declaring as an Environment Variable 
os.environ["MAIN_STORAGE_PATH"] = main_storage_path

# COMMAND ----------

# Read the table
summarised_df = spark.read.table("book_reviews_summarised")

# COMMAND ----------

display(summarised_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregating Summaries

# COMMAND ----------

from pyspark.sql import functions as SF
import re

@SF.udf("string")
def prep_for_condensing(summary_text):
    summary_text = summary_text.split(":\n")[-1]
    summary_text = re.sub(r"\d+\.", "", summary_text)
    summary_text = summary_text.replace("-", " ")
    summary_text = summary_text.replace("\n", "")
    summary_text = summary_text.replace("..", ".")
    summary_text = summary_text.strip()
    summary_text = summary_text.replace("  ", " ")
    return summary_text


agg_summarised_df = (
    summarised_df
    .withColumn("long_review_summary", prep_for_condensing(SF.col("llm_summary")))
    .withColumn("weighted_avg_star_rating", SF.col("avg_star_rating") * SF.col("n_reviews"))
    .groupBy("asin", "title", "author", "week_start", "star_rating_class")
    .agg(
        SF.sum("weighted_avg_star_rating").alias("weighted_avg_star_rating"),
        SF.sum("n_tokens").alias("n_review_tokens"),
        SF.sum("n_reviews").alias("n_reviews"),
        SF.count("*").alias("batch_count"),
        SF.first("llm_summary").alias("review_summary"),
        SF.collect_list("long_review_summary").alias("long_review_summary_array"),
    )
    .withColumn("needs_condensing", SF.col("batch_count") > 1)
    .withColumn("avg_star_rating", SF.round(SF.col("weighted_avg_star_rating") / SF.col("n_reviews"), 2))
    .withColumn(
        "review_summary", 
        SF.when(SF.col("needs_condensing") == False, SF.col("review_summary"))
        .otherwise(SF.concat_ws(" ", SF.col("long_review_summary_array")))
    )
    .drop("weighted_avg_star_rating", "review_summary_array", "long_review_summary_array")
    .orderBy("asin", "title", "author", "week_start", "star_rating_class")
)

display(agg_summarised_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Further Condensing Required

# COMMAND ----------

condense_df = agg_summarised_df.filter(SF.col("needs_condensing") == True)

print("Row count:", condense_df.count())

# COMMAND ----------

# External Imports
from pyspark.sql import functions as SF
import tiktoken

# Function to count tokens using tiktoken
@SF.udf("string")
def truncate_text(text):
    max_token_length = 1800
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = list(encoder.encode(text))
    if len(tokens) > max_token_length:
        text = encoder.decode(tokens[:max_token_length])
    return text

# Build token counter UDF
@SF.udf("int")
def calculate_n_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(string))
    except TypeError:
        # We implement this part to be able to deal with text that cannot be encoded
        num_tokens = -1
    return num_tokens

condense_df = (
    condense_df
    .withColumn("review_summary", truncate_text(SF.col("review_summary"))) # Truncate
    .withColumn("summary_n_tokens", calculate_n_tokens("review_summary")) # Recount
)

display(condense_df.orderBy(SF.col("summary_n_tokens").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Prompts

# COMMAND ----------

positive_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify the five distinct aspects that readers enjoyed about the book. Return the results in five concise bullet points.

Reviews: {review}

### Response:
"""

negative_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify five distinct aspects that readers disliked about the book. Return the results in five concise bullet points.

Reviews: {review}

### Response:
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Instructions

# COMMAND ----------

# External Imports
from pyspark.sql import functions as SF
from pyspark.sql.types import StringType
import pandas as pd

# Build Instruction Builder UDF
@SF.udf("string")
def build_instructions(review, rating_class):
    instructed_review = ""
    if rating_class == "high":
        instructed_review = positive_prompt.format(review=review)
    elif rating_class == "low":
        instructed_review = negative_prompt.format(review=review)
    return instructed_review

# Apply
condense_df = (
    condense_df
    .withColumn(
        "model_instruction",
        build_instructions(SF.col("review_summary"), SF.col("star_rating_class")),
    )
)

display(condense_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Distributed Inference

# COMMAND ----------

# External Imports
from pyspark.sql import functions as SF
import pandas as pd

# Build Inference Function
@SF.pandas_udf("string", SF.PandasUDFType.SCALAR_ITER)
def run_distributed_inference(iterator):

    # External Imports
    from transformers import AutoTokenizer
    import ctranslate2
    import os

    # Define the paths
    local_tokenizer_path = f"{main_storage_path}/mpt-7b-tokenizer"
    local_model_optimised_path = f"{main_storage_path}/mpt-7b-ct2"

    # Params
    temperature = 0.1
    max_new_tokens = 200
    batch_size = 20
    do_sample = True

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the model
    mpt_optimised_model = ctranslate2.Generator(
        model_path=local_model_optimised_path,
        device="cuda",
        device_index=0,
        compute_type="bfloat16"
    )

    for requests in iterator:
        # Encode requests with tokenizer
        batch_tokens = [tokenizer.encode(x) for x in requests.to_list()]
        batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_tokens]

        # Batch results
        batch_results = mpt_optimised_model.generate_batch(
            batch_tokens,
            max_batch_size=batch_size,
            max_length=max_new_tokens,
            include_prompt_in_result=False,
            sampling_temperature=temperature,
        )

        # Batch decode
        decoded_results = [tokenizer.decode(x.sequences_ids[0]) for x in batch_results]

        yield pd.Series(decoded_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Inference

# COMMAND ----------

# Repartition
condense_df = condense_df.repartition(2)

# Run Inference
condense_df = condense_df.withColumn(
    "condensed_review_summary", run_distributed_inference(SF.col("model_instruction"))
)

# Select only the required columns
condense_df = condense_df.select(
    "asin",
    "title",
    "author",
    "week_start",
    "star_rating_class",
    "condensed_review_summary",
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Merge the DFs Back

# COMMAND ----------

agg_summarised_df = (
    agg_summarised_df.drop("condensed_review_summary")
    .join(
        condense_df, 
        how="left", 
        on=["asin", "title", "author", "week_start", "star_rating_class"]
    )
    .withColumn(
        "final_review_summary", 
        SF.when(SF.col("needs_condensing") == True, SF.col("condensed_review_summary"))
        .otherwise(SF.col("review_summary"))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save

# COMMAND ----------

(
    agg_summarised_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("book_reviews_condensed")
)
