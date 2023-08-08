# Databricks notebook source
# MAGIC %md
# MAGIC # Summarisation
# MAGIC We have prepped our data, and it is ready for summarisation. In this notebook, we are going to cover how we can use an LLM to summarise the reviews we have batched.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Globals & Defaults

# COMMAND ----------

# MAGIC %sh
# MAGIC pip freeze | grep transformers

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- You can skip this line if no-UC
# MAGIC USE CATALOG mas;
# MAGIC
# MAGIC -- Sets the standard database to be used in this notebook
# MAGIC USE SCHEMA review_summarisation;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Retrieval

# COMMAND ----------

from pyspark.sql import functions as SF

batched_reviews_df = spark.read.table("batched_book_reviews")

display(
    batched_reviews_df
    .filter(SF.col("star_rating_class")=="high")
    .orderBy("author")
    # .select("concat_review_text")
    .limit(2)
    # .collect()
)

# negative_review_example = (
#     batched_reviews_df
#     .filter(SF.col("star_rating_class")=="low")
#     .select("concat_review_text")
#     .first()[0]
# )

# COMMAND ----------

print(positive_review_examples[0])

# COMMAND ----------

print(positive_review_example)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Loading

# COMMAND ----------

from transformers.utils import logging as t_logging

t_logging.set_verbosity_error()
t_logging.disable_progress_bar()

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# Falcon-7B-Instruct revisions in https://huggingface.co/tiiuae/falcon-7b-instruct/commits/main
model_name = "tiiuae/falcon-7b-instruct"
revision="9f16e66a0235c4ba24e321e3be86dd347a7911a0"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    revision=revision,
    device_map="auto",
)

model.config.use_cache = False

pipeline = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision=revision,
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prompt Engineering

# COMMAND ----------

synth_positive_example = """"This book was a delightful read! From the start, the author grabs your attention with her fluid writing style and beautifully crafted sentences. The main character, Sarah, is someone you can't help but root for. She's strong, resilient, and has a wonderful sense of humor that makes you chuckle every now and then. The author does a brilliant job of making Sarah relatable. I felt like I was right there with her as she navigated through her trials and tribulations. The supporting characters are equally well-written. Each one has their unique quirks and traits that make them memorable. The plot is what makes this book stand out. It's a roller-coaster ride of emotions. There were moments where I was holding my breath, moments where I was laughing, and moments where I was on the verge of tears. The plot twists are unexpected and keep you on your toes. You never know what's going to happen next! I was hooked from start to finish. Another aspect of the book I enjoyed is the world-building. The author creates a magical world that is both enchanting and terrifying. The descriptions are so vivid, it's like you're actually there. I could easily visualize the majestic castles, the dark and eerie forests, and the bustling marketplaces. The author's attention to detail is commendable. The themes explored in this book are thought-provoking. It touches upon topics like bravery, friendship, love, loss, and betrayal. The book is not just a fun and engaging read, but it also makes you think. It shows you that life is not always black and white. There are shades of grey. One of my favorite parts of the book is the romance. It's not overbearing or cliché. It's subtle, sweet, and develops naturally over the course of the book. The chemistry between Sarah and the love interest is undeniable. I was rooting for them throughout the book. The ending of the book was satisfying. It ties up all loose ends while leaving room for a sequel. I am eagerly waiting for the next book in the series. In conclusion, this book is a well-rounded, captivating, and thought-provoking read. It's a page-turner that will keep you hooked from the first page to the last. The characters are lovable, the plot is engaging, the world-building is mesmerizing, and the themes are impactful. I highly recommend this book to all fantasy lovers."""

# COMMAND ----------

import time

positive_prompt_1 = """Input: {review}
Provide a three bullet-point summary capturing what customers liked about this book."""

positive_prompt_2 = """Input: {review}
Identify three aspects that readers liked about the book and provide a summary for each."""

positive_prompt_3 = """Input: {review}
Identify three aspects that readers liked about the book."""

positive_prompt_4 = """Input: {review}
Distill and provide three bullet points capturing what customers most appreciated about the book they reviewed."""

positive_prompt_5 = """Input: Identify three aspects that readers liked about the book.
{review}
Identify three aspects that readers liked about the book."""

positive_prompt_6 = """Input: Identify three aspects that readers liked about the book and provide a summary for each.
{review}
Identify three aspects that readers liked about the book and provide a summary for each."""

positive_prompt_7 = """Input: Identify three distinct and specific aspects that readers enjoyed about the book and provide a bullet point summary for each.
{review}
Identify three distinct and specific aspects that readers enjoyed about the book and provide a bullet point summary for each."""


all_prompts = [
    # positive_prompt_1,
    positive_prompt_2,
    # positive_prompt_3,
    # positive_prompt_4,
    # positive_prompt_5,
    # positive_prompt_6,
    # positive_prompt_7,
]

def timed_generation(prompt):
    # Create request
    request = prompt.format(review=positive_review_examples[0])

    # Record the start time
    start_time = time.time()

    # Generate the response
    response = pipeline(
        request,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Record time elapsed
    finish_time = time.time()
    elapsed_time = round(finish_time - start_time, 2)

    # Parse the response
    response = response[0]["generated_text"].split(request)[-1]

    # Form output
    results = {
        "prompt": prompt,
        "request": request,
        "elapsed_time": elapsed_time,
        "response": response
    }
    return results

all_results = []
for select_prompt in all_prompts:
    single_result = timed_generation(prompt=select_prompt)
    all_results.append(single_result)

# COMMAND ----------

i = 1
for single_result in all_results:
    print("-" * 15)
    print(f"Falcon-7B-Instruct Test {i}")
    print("Prompt:")
    print(single_result["prompt"])
    print("Generation time:", single_result["elapsed_time"], "seconds")
    print("Response:")
    print(single_result["response"])
    print()
    i += 1

# COMMAND ----------

batch_coef = 1

positive_reviews = [positive_review_example] * batch_coef

formatted_requests = [
    positive_prompt_6.format(review=pos_rev) 
    for pos_rev in positive_reviews
]

positive_responses = pipeline(
    formatted_requests,
    max_new_tokens=150,
    temperature=0.2,
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

# basic_example_review = """Love this phone, it's easy to use and has a long-lasting battery. Great camera as well. The tablet I bought was excellent, super fast and the screen is crystal clear. However, the headphones I bought weren't good, they have low audio quality and are uncomfortable."""

# positive_prompt_5 = """Input: From the text below composed of several book reviews, distill and provide three bullet points capturing what customers most appreciated about the book they reviewed.
# {review}"""

# positive_prompt_6 = """Using the text below, identify three aspects that readers liked about the book and provide a summary for each.
# Input: {review}"""

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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Download

# COMMAND ----------


