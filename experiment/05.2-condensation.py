# Databricks notebook source
# MAGIC %md
# MAGIC # Condensation
# MAGIC
# MAGIC We have successfully summarised our reviews in the previous notebook. So why the need for another one ? And what is condensation ?
# MAGIC
# MAGIC This is the part where we can start thinking about how these review summaries are going to be used in a real life scenario. Chances are, there is going to be a product analysis/development team in the company's product department who are going to examine what comes out of the model.
# MAGIC
# MAGIC The good news is, we just saved them a lot of time! The last notebook summarised over 3 million reviews in 4.5 hours.. That is exceptionally good if we were to compare how long it would take for a team of people to do the same, and also - how dreadful they would find the task after a certain point.. 
# MAGIC
# MAGIC However, our job is not yet done, because we want to aim for a scenario where the product team gets to analyse reviews on a weekly basis. What we know is that for some weeks and for some books, we had to batch the reviews, meaning that we ended up with many summaries in a given week.
# MAGIC
# MAGIC Having to read multiple summaries per week defeats the purpose of the project, so now, what we can aim to do is to condense the summaries for these weeks of increased reviews and create almost like a "summary of summaries"
# MAGIC
# MAGIC The flow of this notebook will be similar to the previous one, with some changes to the prompts and data used.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Setup Used:**
# MAGIC
# MAGIC - Runtime: 13.2 ML + GPU
# MAGIC - Cluster:
# MAGIC   - Machine: GPU with > 20GB (For Driver & Worker) 
# MAGIC   - 3+ Workers
# MAGIC   - Recommended GPUs: Nvidia A100 or A10 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Library Installation
# MAGIC
# MAGIC We can start by installing the libraries we are going to need for this work. These are going to be the same with the summarisation notebook.

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

# MAGIC %md
# MAGIC #### Data Defaults
# MAGIC Specifying our data defaults for catalog and schema.

# COMMAND ----------

# Imports
from config import CATALOG_NAME, SCHEMA_NAME

# You can skip this line if no-UC
_ = spark.sql(f"USE CATALOG {CATALOG_NAME};")

# Sets the standard database to be used in this notebook
_ = spark.sql(f"USE SCHEMA {SCHEMA_NAME};")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setting Paths
# MAGIC Specifying the paths we are going to use in this notebook.

# COMMAND ----------

# Import the OS system to declare a ENV variable
from config import CATALOG_NAME, SCHEMA_NAME
import os

# Setting up the storage path (please edit this if you would like to store the data somewhere else)
main_storage_path = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/model_store"

# Declaring as an Environment Variable 
os.environ["MAIN_STORAGE_PATH"] = main_storage_path

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading Data
# MAGIC
# MAGIC Reading the summarised dataframe we created in the last notebook.

# COMMAND ----------

# Read the table
summarised_df = spark.read.table("book_reviews_summarised")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregating Summaries
# MAGIC
# MAGIC As a first step, we need to transform the dataset so we can understand which weeks will need condensation, as well as prepare the summaries which are going to fed into the model.
# MAGIC
# MAGIC We are going to create a UDF for preparing the summaries which will need to be condensed. This UDF is going to turn the summaries in to plain text so that the model can have an easier time processing it because at the moment our summaries have characters like `\n` (new line).
# MAGIC
# MAGIC Then, we are going to decide on which weeks will need condensation based the number of batches in the week. If the number of batches is bigger than 1, the week will be marked for condensation.

# COMMAND ----------

# Imports
from pyspark.sql import functions as SF
import re

# Build UDF for text prep
@SF.udf("string")
def prep_for_condensing(summary_text):
    summary_text = summary_text.split(":\n")[-1]
    summary_text = re.sub(r"\d+\.", "", summary_text)
    summary_text = summary_text.replace("-", " ")
    summary_text = summary_text.replace("\n", ".")
    summary_text = summary_text.replace("..", ".")
    summary_text = summary_text.strip()
    summary_text = summary_text.replace("  ", " ")
    return summary_text

# Build the aggregated dataframe
agg_summarised_df = (
    summarised_df
    # Clean reviews
    .withColumn("long_review_summary", prep_for_condensing(SF.col("llm_summary")))
    # Calculate weighted average so we can get to a weekly average
    .withColumn("weighted_avg_star_rating", SF.col("avg_star_rating") * SF.col("n_reviews"))
    # Group by meta columns
    .groupBy("asin", "title", "author", "week_start", "star_rating_class")
    .agg(
        SF.sum("weighted_avg_star_rating").alias("weighted_avg_star_rating"),
        SF.sum("n_tokens").alias("n_review_tokens"),
        SF.sum("n_reviews").alias("n_reviews"),
        SF.count("*").alias("batch_count"),
        SF.first("llm_summary").alias("review_summary"),
        SF.collect_list("long_review_summary").alias("long_review_summary_array"),
    )
    # Mark weeks that need condensing
    .withColumn("needs_condensing", SF.col("batch_count") > 1)
    # Re-calculate avg star rating on a weekly basis
    .withColumn("avg_star_rating", SF.round(SF.col("weighted_avg_star_rating") / SF.col("n_reviews"), 2))
    # Assing review summary based on condensing requirement
    .withColumn(
        "review_summary", 
        # If False, get the regular summary
        SF.when(SF.col("needs_condensing") == False, SF.col("review_summary"))
        # If True, get the cleaned and concatenated summaries
        .otherwise(SF.concat_ws(". ", SF.col("long_review_summary_array")))
    )
    # Drop unused columns
    .drop("weighted_avg_star_rating", "review_summary_array", "long_review_summary_array")
    .orderBy("asin", "title", "author", "week_start", "star_rating_class")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter Focus Weeks
# MAGIC
# MAGIC Now that we have our dataframe, we can filter it to create a sub dataframe which will hold the focus weeks that require condensation. We can do that by ising the

# COMMAND ----------

# Filter with flag
condense_df = agg_summarised_df.filter(SF.col("needs_condensing") == True)

# Print number of rows (count of weeks that needs condensing)
print("Row count:", condense_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ensure Text Length
# MAGIC
# MAGIC Most of the weeks won't need this, but there can be some extremes where if a book recevied extreme number  reviews in a given week, the token length might become too large for our model to process. For that reason, we can follow a similar flow which we used in the explore & prep notebook to ensure desired token length.
# MAGIC
# MAGIC For this case, we can be more relaxed about our token length and go up to 1800 tokens since the number of examples we are going to have to process is going to be much less.

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

# Apply truncation and count tokens
condense_df = (
    condense_df
    .withColumn("review_summary", truncate_text(SF.col("review_summary"))) # Truncate
    .withColumn("summary_n_tokens", calculate_n_tokens("review_summary")) # Recount
)

# COMMAND ----------

display(
    condense_df
    .groupBy()
    .agg(SF.sum("summary_n_tokens"))
)

# COMMAND ----------

display(condense_df)

# COMMAND ----------

# External Imports
from transformers import AutoTokenizer
import ctranslate2
import os

# Define the paths
local_tokenizer_path = f"{main_storage_path}/mpt-7b-tokenizer"
local_model_optimised_path = f"{main_storage_path}/mpt-7b-ct2"

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

# COMMAND ----------

from pyspark.sql import functions as SF

top_examples = [x[0] for x in (
    condense_df
    .filter(SF.col("star_rating_class") == "high")
    .orderBy(SF.col("summary_n_tokens").desc())
    .limit(5)
    .select("review_summary")
    .collect()
)]


mid_examples = [x[0] for x in (
    condense_df
    .filter(SF.col("star_rating_class") == "high")
    .filter(SF.col("summary_n_tokens") < 1250)
    .orderBy(SF.col("summary_n_tokens").desc())
    .limit(5)
    .select("review_summary")
    .collect()
)]

bottom_examples = [x[0] for x in (
    condense_df
    .filter(SF.col("star_rating_class") == "high")
    .orderBy(SF.col("summary_n_tokens").asc())
    .limit(5)
    .select("review_summary")
    .collect()
)]

all_examples = top_examples + mid_examples + bottom_examples

# COMMAND ----------

print(all_examples[0])

# COMMAND ----------

# Params
temperature = 0.1
max_new_tokens = 250
batch_size = 10
repetition_penalty = 1.05
top_k = 50
top_p = 0.9


prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Given the text below, which consists of a compilation of review summaries about a book, please extract bullet point summaries that highlight what customers specifically liked and appreciated about the book.

Text: {review}.

Return five succint bullet points.

### Response:
"""

def generate_predictions(iterator):
    results = []
    # for requests in iterator:
        # Encode requests with tokenizer
    batch_tokens = [tokenizer.encode(x) for x in iterator]
    batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_tokens]

    # Batch results
    batch_results = mpt_optimised_model.generate_batch(
        batch_tokens,
        max_batch_size=batch_size,
        max_length=max_new_tokens,
        include_prompt_in_result=False,
        sampling_temperature=temperature,
        sampling_topk=top_k,
        sampling_topp=top_p,
        repetition_penalty=repetition_penalty,
    )

    # Batch decode
    decoded_results = [tokenizer.decode(x.sequences_ids[0]) for x in batch_results]

        # results.append(decoded_results[0])
    # return results
    return decoded_results

all_examples_prompted = [prompt_template.format(review=x) for x in all_examples]

condensed_results = generate_predictions(all_examples_prompted)

for request, response in zip(all_examples_prompted, condensed_results):
    print(request[:1000]+"...")
    print("\nReponse:")
    print(response)
    print("\n" + "*"*15 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Prompts
# MAGIC
# MAGIC We can now build the prompts we are going to use for condensation. These prompts are going to be very similar to the ones we used before, except we are going to ask the model.

# COMMAND ----------

positive_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify five distinct aspects that readers enjoyed about the book. Return the result as five succinct bullet points.

Reviews: {review}

### Response:
"""

negative_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify five distinct aspects that readers disliked about the book. Return the result as five succinct bullet points.

Reviews: {review}

### Response:
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Instructions
# MAGIC
# MAGIC Using a UDF, we get to use the prompts we built above and create a model instruction column by putting the summaries in the prompts.

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Distributed Inference
# MAGIC
# MAGIC This part of the code remains almost identical to the one we used in the previous notebook, however one change we make is to reduce the batch size from 20 to 10. This is because we are letting the maximum token length to be 2x than how it was before, so reducing this ensures that we don't run into GPU MEM problems.

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
    batch_size = 5
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
# MAGIC
# MAGIC Applying the inference function we created on our dataframe, and extracting results. Same with the last notebook, don't forget to adjust the repartition count depending on the number of workers you have in your cluster.

# COMMAND ----------

# Imports
from pyspark import SparkContext

# Auto get number of workers
sc = SparkContext.getOrCreate()

# Subtract 1 to exclude the driver
num_workers = len(sc._jsc.sc().statusTracker().getExecutorInfos()) - 1  

# Repartition with respect to number of workers
condense_df = condense_df.repartition(num_workers)

# Run inference
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
# MAGIC ### Merge Condensed with Summarised
# MAGIC
# MAGIC Now that we have condensed our summaries, we can go ahead and merge the condensed summaries back to our main dataframe, and then build a new column to capture them.

# COMMAND ----------

# Join the condense dataframe back to summarised
agg_summarised_df = (
    agg_summarised_df
    .join(
        condense_df, 
        how="left", 
        on=["asin", "title", "author", "week_start", "star_rating_class"]
    )
    # Build a new column for the final result
    .withColumn(
        "final_review_summary",
        # Take the condensed version if it required condensing
        SF.when(SF.col("needs_condensing") == True, SF.col("condensed_review_summary"))
        # Take the regular version otherwise
        .otherwise(SF.col("review_summary"))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Condensed Summaries
# MAGIC
# MAGIC Our work is done, and is ready to be saved.

# COMMAND ----------

(
    agg_summarised_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("book_reviews_condensed")
)
