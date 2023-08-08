# Databricks notebook source
# MAGIC %md
# MAGIC # Summarisation
# MAGIC We have prepped our data, and it is ready for summarisation. In this notebook, we are going to cover how we can use an LLM to summarise the reviews we have batched.

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- You can skip this line if no-UC
# MAGIC USE CATALOG mas;
# MAGIC
# MAGIC -- Sets the standard database to be used in this notebook
# MAGIC USE SCHEMA review_summarisation;

# COMMAND ----------

from pyspark.sql import functions as SF

batched_reviews_df = spark.read.table("batched_book_reviews")

positive_review_example = (
    batched_reviews_df
    .filter(SF.col("star_rating_class")=="high")
    .select("concat_review_text")
    .first()[0]
)

negative_review_example = (
    batched_reviews_df
    .filter(SF.col("star_rating_class")=="low")
    .select("concat_review_text")
    .first()[0]
)

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


import os
os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['TRANSFORMERS_CACHE'] = '/local_disk0/hf'

tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga-7B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("stabilityai/StableBeluga-7B", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", use_cache=False)
system_prompt = "### System:\nYou are StableBeluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

message = "Write me a poem please"
prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

print(tokenizer.decode(output[0], skip_special_tokens=True))


# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of falcon-7b-instruct in https://huggingface.co/tiiuae/falcon-7b-instruct/commits/main
model_name = "stabilityai/StableBeluga-7B"
revision="329adcfc39f48dce183eb0b155b732dbe03c6304"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    padding_side="left", 
    use_fast=False
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # revision=revision,
    device_map="auto",
)


pipeline = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    revision=revision,
)

# Required tokenizer setting for batch inference
# pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

basic_example_review = """Love this phone, it's easy to use and has a long-lasting battery. Great camera as well. The tablet I bought was excellent, super fast and the screen is crystal clear. However, the headphones I bought weren't good, they have low audio quality and are uncomfortable."""

# COMMAND ----------

positive_prompt = """Input: Read the following combined book reviews and provide a bullet-point summary capturing what the customers liked about this book:

{review}

Provide a three bullet-point summary capturing what customers liked about this book"""

positive_prompt_2 = """Input: {review}
Identify three aspects that readers liked about the book and provide a summary for each."""

# COMMAND ----------

positive_prompt_3 = """Input: {review}
Identify three aspects that readers liked about the book."""

batch_coef = 1

positive_reviews = [positive_review_example] * batch_coef

formatted_requests = [
    positive_prompt_3.format(review=pos_rev) 
    for pos_rev in positive_reviews
]

positive_responses = pipeline(
    formatted_requests,
    max_new_tokens=150,
    temperature=0.4,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    batch_size=1
)

for response, f_request, r_request in zip(positive_responses, formatted_requests, positive_reviews):
    _response = response[0]["generated_text"].split(f_request)[-1]
    print("-"*10)
    # print(f"Review:\n{r_request}")
    print(f"\nFalcon-7B:\n{_response}")

# COMMAND ----------

negative_prompt = """Input: Read the following combined book reviews and provide a bullet-point summary capturing what the customers didn't like about this book:

{review}

Provide a three bullet-point summary capturing what customers didn't like about this book"""

response = pipeline(
    negative_prompt.format(review=negative_review_example),
    max_new_tokens=150,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

print(response[0]["generated_text"])
