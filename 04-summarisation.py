# Databricks notebook source
# MAGIC %md
# MAGIC # Summarisation
# MAGIC
# MAGIC Our instructions are ready and the reviews are waiting to be summarised! We can now take the necessary steps to begin our inference (summirisation task).
# MAGIC
# MAGIC Before we do so, it might help to do a couple of things.. We want to optimise the speed of inference as much as possible (without trading off quality) and we also want to distribute our inference so we can scale properly. 
# MAGIC
# MAGIC In this notebook, we will cover the optimisations that can be done pre-summarisation, and how we can paralleze the work.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Up
# MAGIC
# MAGIC We can start by installing the libraries we are going to need for this work. As always, you can choose to specify these using the cluster's configuration page so that the cluster can auto spawn with these libraries installed. Another benefit - the libraries stay there even if you detach from the notebook (which won't be the case here..)

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
# MAGIC ### Globals
# MAGIC We can now specify our data defaults, and the paths we are going to use in this notebook..

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- You can skip this line if no-UC
# MAGIC USE CATALOG mas;
# MAGIC
# MAGIC -- Sets the standard database to be used in this notebook
# MAGIC USE SCHEMA review_summarisation;

# COMMAND ----------

# Import the OS system to declare a ENV variable
import os

# Setting up the storage path (please edit this if you would like to store the data somewhere else)
main_storage_path = "/Volumes/mas/review_summarisation/model_store"

# Declaring as an Environment Variable 
os.environ["MAIN_STORAGE_PATH"] = main_storage_path

# Set local model paths
local_model_path = f"{main_storage_path}/mpt-7b-instruct"
local_tokenizer_path = f"{main_storage_path}/mpt-7b-tokenizer"
local_model_optimised_path = f"{main_storage_path}/mpt-7b-ct2"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Retrieval
# MAGIC
# MAGIC We created the batched instructions dataset in the last noteboook, which was produced after our prompt engineering tests. This dataset includes a `model_instruction` column, which has the text we are going to feed to the LLM with the instructions.

# COMMAND ----------

# Read the instructions dataframe
instructions_df = spark.read.table("batched_instructions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inferecene Optimisations
# MAGIC
# MAGIC Lets see if we can optimise the speed of our inference.. There is a library called `CTranslate2` which can take existing transformer like models, and optimise them by inference. This can help us greatly, and reduce the resources we may need to use.
# MAGIC
# MAGIC The library works by converting an existing transformer into a generator. Which essentially has the same properties, but with some added options. 
# MAGIC
# MAGIC This library offers quantisation as well.. Quantisation helps with making the model run with a smaller footprint on the GPU. However, it comes with a trade-off - the answer quality begins to drop as you quantise further. 
# MAGIC
# MAGIC But, for some cases it might still make sense. If you would like use a more performant quantisation, you can definitely lower it here. 

# COMMAND ----------

# External Imports
from ctranslate2.converters import TransformersConverter

# Initiate the converter
if os.path.isdir(local_model_optimised_path):
    print("Optimised model exists")
else:
    mpt_7b_converter = TransformersConverter(
        model_name_or_path=local_model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Request conversion
    mpt_7b_converter.convert(
        output_dir=local_model_optimised_path,
        quantization="bfloat16"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Converted Model & Tokenizer
# MAGIC
# MAGIC Our model has been converted and is now ready to do be tested for inference. Let's load it up with the tokenizer, and see what we can do..

# COMMAND ----------

# External Imports
from transformers import AutoTokenizer
import ctranslate2
import os
import time

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

# MAGIC %md
# MAGIC ### Test Flow Build
# MAGIC
# MAGIC We can build a test flow to see how the model does, and experiemnt with some parameters. We especially want to focus on the batch size parameter here to find it's sweet spot.

# COMMAND ----------

def run_inference(requests, batch_size):
    
    # Create a return dict
    return_dict = {}

    # Time
    encoding_start = time.time()

    # Encode requests with tokenizer
    batch_tokens = [tokenizer.encode(x) for x in requests]
    batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_tokens]

    # Time
    return_dict["encoding_time"] = round(time.time() - encoding_start, 4)
    generation_start = time.time()

    # Generate results with the model
    batch_results = mpt_optimised_model.generate_batch(
        batch_tokens,
        max_batch_size=batch_size,
        max_length=150,
        include_prompt_in_result=False,
        sampling_temperature=0.1,
    )
    
    # Time
    return_dict["generation_time"] = round(time.time() - generation_start, 4)
    decoding_start = time.time()

    # Decode results with the tokenizer
    decoded_results = [tokenizer.decode(x.sequences_ids[0]) for x in batch_results]

    # Time
    return_dict["decoding_time"] = round(time.time() - decoding_start, 4)
    return_dict["total_time"] = round(time.time() - encoding_start, 4)

    # Prepare and Return
    return_dict["results"] = decoded_results
    return return_dict

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieving few examples from our dataset here so we can do some tests

# COMMAND ----------

# Random sample examples
examples = (
    instructions_df
    .sample(False, 0.01, seed=42)
    .select("model_instruction")
    .limit(200)
    .collect()
)
examples = [x[0] for x in examples]

# COMMAND ----------

# MAGIC %md
# MAGIC The code below can help us with identifying the optimal spot for the batch size parameter

# COMMAND ----------

# Batch sizes to be tested
batch_size_test = [5, 10, 15, 20, 25, 30, 35]

# Speed Test
for batch_test in batch_size_test:
    _result = run_inference(examples, batch_size=batch_test)
    print("-"*15)
    print("Batch Size", batch_test)
    print(_result["total_time"])

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like 20 is the number we are looking for. Let see how the results look like when we use this parameter

# COMMAND ----------

results = run_inference(examples, batch_size=20)

for key in results.keys():
    if "time" in key:
        print(f"{key}: {results[key]}")

for _request, _response in zip(examples, results["results"]):
    print("-" * 15)
    print(_request)
    print()
    print(_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Inference
# MAGIC
# MAGIC Now that our test flow works and we have a batch size we can use, lets build a similar flow, but this time to be distributed across a cluster..

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
    max_new_tokens = 150
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

def sample_by_row_count(df, row_count):
    df_count = df.count()
    required_fraction = round(row_count / df_count, 4)
    df = df.sample(False, fraction=required_fraction, seed=42)
    return df

# Sample
# test_df = sample_by_row_count(instructions_df, 1000)
test_df = instructions_df

# Repartition
test_df = test_df.repartition(2)

# Run Inference
test_df = (
    test_df
    .withColumn("llm_summary", run_distributed_inference(SF.col("model_instruction")))
)

# Save
(
    test_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("book_reviews_summarised")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM predictions_test_tmp

# COMMAND ----------

# External Imports
import ctranslate2

# Load the optimised model
mpt_optimised_model = ctranslate2.Generator(
    model_path=local_model_optimised_path,
    device="cuda",
    device_index=0,
    # inter_threads=10,
    # intra_threads=2
)

# COMMAND ----------

import time


# Build an optimised piepline
def optimised_mpt_pipeline(requests):
    model_responses = []
    for single_request in requests:
        # Encode Requests
        tokenized_request = tokenizer.encode(single_request)
        # Convert IDs to Tokens
        tokenized_request = tokenizer.convert_ids_to_tokens(tokenized_request)
        # Generate response with optimised model
        tokenized_result = mpt_optimised_model.generate_batch(
            [tokenized_request],
            max_batch_size=0,
            max_length=100,
            include_prompt_in_result=False,
            sampling_temperature=0.2,
        )
        # Decode Results
        decoded_result = tokenizer.decode(tokenized_result[0].sequences_ids[0])
        # Append to responses
        model_responses.append(decoded_result)
    return model_responses


def optimised_parallel_mpt_pipeline(requests, batch_size):
    
    # Create a return dict
    return_dict = {}

    encoding_start = time.time()
    # Batch Encode requests
    batch_encoded = tokenizer.batch_encode_plus(
        requests, padding=True, truncation=True, return_tensors="pt"
    ).to("cuda")

    # Convert token IDs back to tokens for all sentences
    batch_tokens = [
        tokenizer.convert_ids_to_tokens(ids) for ids in batch_encoded["input_ids"]
    ]
    return_dict["encoding_time"] = round(time.time() - encoding_start, 4)
    
    generation_start = time.time()
    # Batch results
    batch_results = mpt_optimised_model.generate_batch(
        batch_tokens,
        max_batch_size=batch_size,
        max_length=150,
        include_prompt_in_result=False,
        sampling_temperature=0.2,
    )
    return_dict["generation_time"] = round(time.time() - generation_start, 4)
    
    decoding_start = time.time()
    # Batch decode
    decoded_results = [tokenizer.decode(x.sequences_ids[0]) for x in batch_results]

    return_dict["decoding_time"] = round(time.time() - decoding_start, 4)
    return_dict["total_time"] = round(time.time() - encoding_start, 4)

    return_dict["results"] = decoded_results
    return return_dict

# COMMAND ----------

pos_test_responses = optimised_parallel_mpt_pipeline(
    [x.format(review=positive_review_examples[0]) for x in all_positive_prompts]*30,
    batch_size=40
)

for key in pos_test_responses.keys():
    if "time" in key:
        print(key, pos_test_responses[key])

# COMMAND ----------

# tokenizer.pad_token = tokenizer.eos_token

# test_tokens = tokenizer(formatted_requests, return_tensors="pt", padding=True, truncation=True)

# Assuming 'formatted_requests' is a list of sentences
# formatted_requests = ["Sentence 1", "Sentence 2", "Sentence 3"]

# # Encode all sentences together in a batch
# batch_encoded = tokenizer.batch_encode_plus(
#     formatted_requests, padding=True, truncation=True, return_tensors="pt"
# )

# # Convert token IDs back to tokens for all sentences
# test_tokens = [
#     tokenizer.convert_ids_to_tokens(ids) for ids in batch_encoded["input_ids"]
# ]


# test_tokens = tokenizer.encode(formatted_requests[0])
# test_tokens = tokenizer.convert_ids_to_tokens(test_tokens)

# Encode all sentences together in a batch
batch_encoded = tokenizer.batch_encode_plus(formatted_requests, padding=True, truncation=True, return_tensors='pt')

# Convert token IDs back to tokens for all sentences
test_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_encoded['input_ids']]

test_results = mpt_optimised_model.generate_batch(
    test_tokens,
    max_batch_size=10,
    # end_token=tokenizer.eos_token_id,
    max_length=100,
    include_prompt_in_result=False,
    sampling_temperature=0.2,
)

test_results = [tokenizer.decode(x.sequences_ids[0]) for x in  test_results]
# test_text = tokenizer.decode(test_results[0].sequences_ids[0])

# print(test_text)

# COMMAND ----------

    # def _batch_generate(iterator):
    #     # Batch Encode requests
    #     batch_encoded = tokenizer.batch_encode_plus(
    #         requests.to_list(), padding=True, truncation=True, return_tensors="pt"
    #     ).to("cuda")

    #     # Convert token IDs back to tokens for all sentences
    #     batch_tokens = [
    #         tokenizer.convert_ids_to_tokens(ids) for ids in batch_encoded["input_ids"]
    #     ]

    #     # Batch results
    #     batch_results = mpt_optimised_model.generate_batch(
    #         batch_tokens,
    #         max_batch_size=batch_size,
    #         max_length=max_new_tokens,
    #         include_prompt_in_result=False,
    #         sampling_temperature=temperature,
    #     )

    #     # Batch decode
    #     decoded_results = [tokenizer.decode(x.sequences_ids[0]) for x in batch_results]
    #     return decoded_results
