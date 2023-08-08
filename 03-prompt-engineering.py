# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Engineering
# MAGIC We have prepped our data, and it is ready for summarisation. In this notebook, we are going to cover how we can create prompts to use with an LLM to summarise the reviews we have batched.
# MAGIC
# MAGIC There many, many models which we can pick and choose from in the open source community. Huggingface hosts most of these on their hub, and the great thing is, they have more or less standardised the way to interact with these models, so we can use most of them with slight changes to our code.
# MAGIC
# MAGIC So, what should we pay attention to while choosing our model ?
# MAGIC
# MAGIC First, lets talk about model size. When you hear about a LLM, you would usually see that it comes with a parameter value. This can be somehting like 7 Billion parameters, 13, 30, 70, etc.. What does this mean ? This value tells us about how many configurable/trainable parameters a model has, which can tell us about its capacity to understand things. The bigger the model, the more complex tasks it can complete. However, as their size increase, so does their computation requirements: they tend to require more resources to operate. Therefore, picking the smallest size that can do the work is the best way to start. In our case, summarisation can be done by 7B models, so we are going to option for those.
# MAGIC
# MAGIC Then, we need to pick a model which can follow instructions. What does that mean ? Lets take a look at two examples:
# MAGIC
# MAGIC * [MPT-7B-Base Model](https://huggingface.co/mosaicml/mpt-7b)
# MAGIC * [MPT-7B-Instruct Model](https://huggingface.co/mosaicml/mpt-7b-instruct)
# MAGIC
# MAGIC Mosaic's 7B Base model is a pre-trained model, however it has not been fine-tuned for a specific task. It would be a great candidate if we wanted to fine-tune it for a specific task which we have training data for. Where as the Instruct model has been trained on a Instructions dataset, and is more ready to follow instructions. 
# MAGIC
# MAGIC Given this, selecting the instruct model makes more sense, which we will cover in this notebook.
# MAGIC
# MAGIC Lets begin!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Up
# MAGIC Our model requires some specific libraries to be present in the runtime. We can install them using the commands below. This is definitely a good way to start, however, if you are thinking about setting up a cluster which you will continously use with a model as such, another good way to approach library installation can be to specify the libraries you want within your cluster's configuration page.
# MAGIC
# MAGIC For this part of project, we are going to need a GPU enabled instance.
# MAGIC
# MAGIC Our set up for this notebook is:
# MAGIC * Runtime: DBR 13.2 ML + GPU
# MAGIC * Compute (Single Node): `NC25ads_A100_v4`

# COMMAND ----------

# MAGIC %sh
# MAGIC # Check out our driver's GPU
# MAGIC nvidia-smi

# COMMAND ----------

# Install libraries
%pip install -q flash-attn
%pip install -q xformers
%pip install -q torch==2.0.1
%pip install -q triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python

# Restart Python Kernel
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Globals & Defaults
# MAGIC
# MAGIC Lets set the global variables and the default catalogs/schemas we are going to use in this notebook.

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

# Imports
from pyspark.sql import functions as SF

# Read our main dataframe
batched_reviews_df = spark.read.table("batched_book_reviews")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Loading
# MAGIC
# MAGIC We have select the [MPT-7B-Instruct Model](https://huggingface.co/mosaicml/mpt-7b-instruct) model for this specific task, which was built by our friends from [Mosaic ML](https://www.mosaicml.com/). The model it self is very robust, has good performance and can take on summarisation tasks easily. It features a 30B parameter version as well, but that would probably be an overkill for our use case.
# MAGIC
# MAGIC Some other 7B models you can also check out are:
# MAGIC * [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)
# MAGIC * [Llama-2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
# MAGIC * [Stable-Beluga-7B](https://huggingface.co/stabilityai/StableBeluga-7B)
# MAGIC
# MAGIC
# MAGIC We are now going to use the the transformers library from hugging face to download & load the model.
# MAGIC
# MAGIC In the first cell, we will download the tokenizer and the model to a local location. This step is not a deal breaker, but it speeds up the loading process of the model since it makes it so that we don't have to download it each time we need to use it.
# MAGIC
# MAGIC In the cell following that, we will load the downloaded snapshots to the GPU and instantiate a pipeline which will encapsulate the tokenizer and the model to help with text generation.

# COMMAND ----------

# External Imports
from huggingface_hub.utils import (
    logging as hf_logging,
    disable_progress_bars as hfhub_disable_progress_bar,
)
from huggingface_hub import snapshot_download
import os

# Turn Off Info Logging for Transfomers
hf_logging.set_verbosity_error()
hfhub_disable_progress_bar()

# MPT-7B-Instruct revisions in https://huggingface.co/mosaicml/mpt-7b-instruct/commits/main
model_name = "mosaicml/mpt-7b-instruct"
model_revision_id = "925e0d80e50e77aaddaf9c3ced41ca4ea23a1025"
toknizer_name = "EleutherAI/gpt-neox-20b"
tokenizer_revision_id = "9369f145ca7b66ef62760f9351af951b2d53b77f"

# Download the model
local_model_path = f"{main_storage_path}/mpt-7b-instruct/"
if os.path.isdir(local_model_path):
    print("Local model exists")
else:
    print(f"Downloading model to {local_model_path}")
    model_download = snapshot_download(
        repo_id=model_name,
        revision=model_revision_id,
        local_dir=local_model_path,
        local_dir_use_symlinks=False,
    )

# Download the tokenizer
local_tokenizer_path = f"{main_storage_path}/mpt-7b-tokenizer/"
if os.path.isdir(local_tokenizer_path):
    print("Local tokenizer exists")
else:
    print(f"Downloading tokenizer to {local_tokenizer_path}")
    tokenizer_download = snapshot_download(
        repo_id=toknizer_name,
        revision=tokenizer_revision_id,
        local_dir=local_tokenizer_path,
        local_dir_use_symlinks=False,
    )

# COMMAND ----------

# External Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.utils import logging as t_logging
import transformers
import torch

# Turn Off Info Logging for Transfomers
t_logging.set_verbosity_error()
t_logging.disable_progress_bar()

# MPT-7b-instruct revisions in https://huggingface.co/mosaicml/mpt-7b-instruct/commits/main
model_name = local_model_path
toknizer_name = local_tokenizer_path

# Load model's config
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.attn_config["attn_impl"] = "triton"
config.init_device = "cuda:0"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(toknizer_name, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Build the pipeline
mpt_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    config=config,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device=0,
)

# # Required tokenizer setting for batch inference
mpt_pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# MAGIC %sh
# MAGIC # Check out the GPU again to see how the memory has changed (since we loaded the model)
# MAGIC nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC #### Suggested Prompt Template
# MAGIC
# MAGIC Now that we have loaded the model, we can start asking some questions..
# MAGIC
# MAGIC The MPT 7B Instruct model has been fine tuned with a [specific prompt template](https://huggingface.co/mosaicml/mpt-7b-instruct#formatting). You can think about the prompt template as the "right way to ask a question to the model". On the model's webpage, they specfically note that the model should be instructed in this way. Lets take a look at how we can achieve this prompt

# COMMAND ----------

# Suggest template prompt
mpt_template_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{question}
### Response:
"""

# Example Question
example_question = "When does summer start ?"

print(mpt_template_prompt.format(question=example_question))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Few Examples
# MAGIC
# MAGIC Lets run through a few examples to see how our model works..

# COMMAND ----------

# Example requests
requests = [
    "How many days are there in a week ?",
    "When does summer start ?",
    "If you could learn a programming language, which one would you go for ?",
    "What's a Large Language Model ?",
    "What does summarisation mean ?",
    "How can you deal with an angry customer ?"
]

# Format the requests with the prompt teamplte
formatted_requests = [
    mpt_template_prompt.format(question=single_request) 
    for single_request in requests
]

# Generate response
llm_responses = mpt_pipeline(
    formatted_requests, 
    max_new_tokens=200, # How long can the answer be ?
    do_sample=True,
    temperature=0.4, # How creative the model can be ? (1 = max creativity)
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# Print Responses
for response, f_request, r_request in zip(llm_responses, formatted_requests, requests):
    print("-"*10)
    print("User:   ",  r_request)
    print("MPT-7B: ", response[0]["generated_text"].split(f_request)[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC Answers look pretty good overall! 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prompt Engineering
# MAGIC
# MAGIC Now, lets start focussing on our task by taking a look at prompt engineering and why its important..
# MAGIC
# MAGIC The question/instruction we write within the prompt is essentially the task description for model. You can think of this as writing code in English for our model. The more complex the task gets, the more computationally intensive it becomes for the model to answer. And, each model has a different way it likes to be called.
# MAGIC
# MAGIC For example, some models do better when the input text, in our case reviews, is put before the instruction. The best way to understand all of this is through experimentation. Lets take a look at how we set up one here to see which prompts work best for our case. But first, we will need some examples to test with. Lets get a random positive review and a negative review.

# COMMAND ----------

# Retrieve positive examples
positive_review_examples = (
    batched_reviews_df
    .filter(SF.col("star_rating_class")=="high")
    .sample(False, fraction=0.01, seed=42)
    .select("concat_review_text")
    .limit(10)
    .collect()
)
positive_review_examples = [x[0] for x in positive_review_examples]

# Retrieve negative examples
negative_review_examples = (
    batched_reviews_df
    .filter(SF.col("star_rating_class")=="low")
    .sample(False, fraction=0.01, seed=42)
    .select("concat_review_text")
    .limit(10)
    .collect()
)
negative_review_examples = [x[0] for x in negative_review_examples]

# COMMAND ----------

print("Positive Example")
print(positive_review_examples[0][:150] + "...")
print("\n" + "-" * 15 + "\n")
print("Negative Example")
print(negative_review_examples[0][:150] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Positive Prompt Variations
# MAGIC
# MAGIC Starting with our positive examples, here are some prompts we can test:

# COMMAND ----------

# Prompt Variations
positive_prompt_1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Provide three bullet-point summary capturing what customers liked about this book using the reviews below.

Reviews: {review}

### Response:
"""

positive_prompt_2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Identify three aspects that readers liked about the book and provide a summary for each from the reviews below.

Reviews: {review}

### Response:
"""

positive_prompt_3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Distill and provide three bullet points capturing what customers most appreciated about the book from the reviews below.

Reviews: {review}

### Response:
"""

positive_prompt_4 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Identify three distinct and specific aspects that readers enjoyed about the book from the reviews below, and provide a bullet point summary for each.

Reviews: {review}

### Response:
"""

positive_prompt_5 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify three distinct aspects that readers enjoyed about the book. Return the result as three succinct bullet points.

Reviews: {review}

### Response:
"""

positive_prompt_6 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify three distinct aspects that readers enjoyed about the book. Be sure to include any character dynamics, plot elements, or emotional responses mentioned by the reviewers. Return the result as three succinct bullet points.

Reviews: {review}

### Response:
"""

positive_prompt_7 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the provided book reviews and identify distinct aspects in three bullet points that readers enjoyed about the book. For each aspect, provide a brief explanation using the specific details mentioned in the reviews, focusing on character dynamics, plot elements, or emotional responses elicited.

Reviews: {review}

### Response:
"""

# Build a prompts list
all_positive_prompts = [
    positive_prompt_1,
    positive_prompt_2,
    positive_prompt_3,
    positive_prompt_4,
    positive_prompt_5,
    positive_prompt_6,
    positive_prompt_7,
]

# COMMAND ----------

# MAGIC %md
# MAGIC All of them are slifghtly deifferent than each other. We preserve the generic template that was suggested in the model's webpage, however we alter the instruction slightly with each to see how it differs.
# MAGIC
# MAGIC Now, lets write some code to see how each differs.
# MAGIC
# MAGIC One thing to note is that we are reducing the temparature of our model here.. Why ? Because we want to it to be less creative and we want it to follow the instructions better. 

# COMMAND ----------

# External Imports
import time

# Create a function for assesment
def timed_generation(prompt, review_example):
    # Feed our example to the prompt
    request = prompt.format(review=review_example)

    # Record the start time
    start_time = time.time()

    # Generate the response
    response = mpt_pipeline(
        request,
        max_new_tokens=150,
        temperature=0.1,
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
        "review": review_example,
        "request": request,
        "elapsed_time": elapsed_time,
        "response": response
    }
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC Generate summaries using the wide range of prompts

# COMMAND ----------

all_results = []

# For each prompt, try out an example and return results
for select_prompt in all_positive_prompts:
    single_result = timed_generation(
        prompt=select_prompt, 
        review_example=positive_review_examples[0]
    )
    all_results.append(single_result)

i = 1
for single_result in all_results:
    print("-" * 15)
    print(f"MPT-7B-Instruct Test {i}\n")
    print("Prompt:")
    print(single_result["prompt"])
    print(single_result["response"])
    print("\nGeneration time:", single_result["elapsed_time"], "seconds")
    print()
    i += 1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Multiple Positive Review Testing
# MAGIC
# MAGIC As we can see from above, each prompt pushes the model to respond in a different way. Some of them make the model respond with succint answers, where as others make it exolain with more details...
# MAGIC
# MAGIC The generation time recorderded is also a good metric to keep an eye out for. This can tell us how computainatally intensive our task is going to be, and its always to consider that too. The longer it gets, the more resources we are going to have to use to summarise. So, if the short answers suffice, we could option for those and get to spend less resources. However, if we want more details, then we can choose the performance trade-off.
# MAGIC
# MAGIC For this use case, I think Prompt 5 can work nicely.. It goes into details, but not as much as 6, and still can get things done quickly as well.
# MAGIC
# MAGIC Another test could be to see how it does with a variety of examples:

# COMMAND ----------

variety_review_results = []
for select_review in positive_review_examples:
    single_result = timed_generation(
        prompt=positive_prompt_5, 
        review_example=select_review
    )
    variety_review_results.append(single_result)

i = 1
for single_result in variety_review_results:
    print("-" * 15)
    print(f"MPT-7B-Instruct Test {i}\n")
    print("Review:")
    print(single_result["review"][:350] + "...")
    print("\nResponse:")
    print(single_result["response"])
    print("\nGeneration time:", single_result["elapsed_time"], "seconds")
    print()
    i += 1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Negative Prompt Forming
# MAGIC
# MAGIC Following the same flow for the negative prompt, where we want to asses negative reviews and understand what custoemrs dislikedm, and how we can improve the product..

# COMMAND ----------

negative_prompt_1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Provide a three bullet-point summary capturing what customers disliked about this book using the reviews below.

Reviews: {review}

### Response:
"""

negative_prompt_2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Identify three aspects that readers disliked about the book and provide a summary for each from the reviews below.

Reviews: {review}

### Response:
"""

negative_prompt_3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Distill and provide three bullet points capturing what customers most criticized about the book from the reviews below.

Reviews: {review}

### Response:
"""

negative_prompt_4 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Identify three distinct and specific aspects that readers did not enjoy about the book from the reviews below, and provide a bullet point summary for each.

Reviews: {review}

### Response:
"""

negative_prompt_5 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify three distinct aspects that readers disliked about the book. Return the result as three succinct bullet points.

Reviews: {review}

### Response:
"""

negative_prompt_6 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the provided book reviews and identify three distinct aspects that readers disliked about the book. Be sure to include any character dynamics, plot elements, or emotional responses mentioned by the reviewers that led to negative experiences. Return the answer as three succinct bullet points.

Reviews: {review}

### Response:
"""

negative_prompt_7 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the provided book reviews and identify distinct aspects in three bullet points that readers disliked about the book. For each aspect, provide a brief explanation using the specific details mentioned in the reviews, focusing on character dynamics, plot elements, or emotional responses elicited.

Reviews: {review}

### Response:
"""

# Build a full list of prompts
all_negative_prompts = [
    negative_prompt_1,
    negative_prompt_2,
    negative_prompt_3,
    negative_prompt_4,
    negative_prompt_5,
    negative_prompt_6,
    negative_prompt_7,
]

# COMMAND ----------

all_negative_results = []

# For each prompt, try out an example and return results
for select_prompt in all_negative_prompts:
    single_result = timed_generation(
        prompt=select_prompt, 
        review_example=negative_review_examples[0]
    )
    all_negative_results.append(single_result)

i = 1
for single_result in all_negative_results:
    print("-" * 15)
    print(f"MPT-7B-Instruct Test {i}\n")
    print("Prompt:")
    print(single_result["prompt"])
    print(single_result["response"])
    print("\nGeneration time:", single_result["elapsed_time"], "seconds")
    print()
    i += 1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Negative Responses for Different Reviews

# COMMAND ----------

variety_negative_review_results = []
for select_review in negative_review_examples:
    single_result = timed_generation(
        prompt=negative_prompt_5, 
        review_example=select_review
    )
    variety_negative_review_results.append(single_result)

i = 1
for single_result in variety_negative_review_results:
    print("-" * 15)
    print(f"MPT-7B-Instruct Test {i}\n")
    print("Review:")
    print(single_result["review"][:350] + "...")
    print("\nResponse:")
    print(single_result["response"])
    print("\nGeneration time:", single_result["elapsed_time"], "seconds")
    print()
    i += 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding Prompts to Data
# MAGIC
# MAGIC Now that we have selected our prompts, we can go ahead and add these to our dataset, and save, so that we can use them with ease later on in the summarisation part.
# MAGIC
# MAGIC First, lets start by declaring our selections:

# COMMAND ----------

# Define the selected positive prompt
selected_postive_prompt = positive_prompt_5

# Define the selected negative prompt
selected_negative_prompt = negative_prompt_5

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we can go ahead and create a UDF which will take these into account and format our text as we like

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
        instructed_review = selected_postive_prompt.format(review=review)
    elif rating_class == "low":
        instructed_review = selected_negative_prompt.format(review=review)
    return instructed_review

# Apply
batched_instructions_df = (
    batched_reviews_df
    .withColumn(
        "model_instruction",
        build_instructions(SF.col("concat_review_text"), SF.col("star_rating_class")),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Save

# COMMAND ----------

# Save Raw Reviews
(
    batched_instructions_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("batched_instructions")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## TO BE DELETED ONCE SAFE

# COMMAND ----------

# synth_positive_example = """"This book was a delightful read! From the start, the author grabs your attention with her fluid writing style and beautifully crafted sentences. The main character, Sarah, is someone you can't help but root for. She's strong, resilient, and has a wonderful sense of humor that makes you chuckle every now and then. The author does a brilliant job of making Sarah relatable. I felt like I was right there with her as she navigated through her trials and tribulations. The supporting characters are equally well-written. Each one has their unique quirks and traits that make them memorable. The plot is what makes this book stand out. It's a roller-coaster ride of emotions. There were moments where I was holding my breath, moments where I was laughing, and moments where I was on the verge of tears. The plot twists are unexpected and keep you on your toes. You never know what's going to happen next! I was hooked from start to finish. Another aspect of the book I enjoyed is the world-building. The author creates a magical world that is both enchanting and terrifying. The descriptions are so vivid, it's like you're actually there. I could easily visualize the majestic castles, the dark and eerie forests, and the bustling marketplaces. The author's attention to detail is commendable. The themes explored in this book are thought-provoking. It touches upon topics like bravery, friendship, love, loss, and betrayal. The book is not just a fun and engaging read, but it also makes you think. It shows you that life is not always black and white. There are shades of grey. One of my favorite parts of the book is the romance. It's not overbearing or cliché. It's subtle, sweet, and develops naturally over the course of the book. The chemistry between Sarah and the love interest is undeniable. I was rooting for them throughout the book. The ending of the book was satisfying. It ties up all loose ends while leaving room for a sequel. I am eagerly waiting for the next book in the series. In conclusion, this book is a well-rounded, captivating, and thought-provoking read. It's a page-turner that will keep you hooked from the first page to the last. The characters are lovable, the plot is engaging, the world-building is mesmerizing, and the themes are impactful. I highly recommend this book to all fantasy lovers."""

# COMMAND ----------

# import time
# positive_prompt_1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Provide three bullet-point summary capturing what customers liked about this book using the reviews below.

# Reviews: {review}

# ### Response:
# """

# positive_prompt_2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Identify three aspects that readers liked about the book and provide a summary for each from the reviews below.

# Reviews: {review}

# ### Response:
# """

# positive_prompt_3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Distill and provide three bullet points capturing what customers most appreciated about the book from the reviews below.

# Reviews: {review}

# ### Response:
# """

# positive_prompt_4 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Identify three distinct and specific aspects that readers enjoyed about the book from the reviews below, and provide a bullet point summary for each.

# Reviews: {review}

# ### Response:
# """

# positive_prompt_5 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Analyze the provided book reviews and identify three distinct aspects that readers enjoyed about the book. Be sure to include any character dynamics, plot elements, or emotional responses mentioned by the reviewers.

# Reviews: {review}

# ### Response:
# """

# positive_prompt_6 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Analyze the provided book reviews and identify three distinct aspects in bullet points that readers enjoyed about the book. For each aspect, provide a brief explanation using the specific details mentioned in the reviews, focusing on character dynamics, plot elements, or emotional responses elicited.

# Reviews: {review}

# ### Response:
# """



# all_prompts = [
#     # positive_prompt_1,
#     # positive_prompt_2,
#     # positive_prompt_3,
#     # positive_prompt_4,
#     # positive_prompt_5,
#     positive_prompt_6,
# ]


# def timed_generation(prompt):
#     # Create request
#     request = prompt.format(review=positive_review_examples[0])

#     # Record the start time
#     start_time = time.time()

#     # Generate the response
#     response = mpt_pipeline(
#         request,
#         max_new_tokens=150,
#         temperature=0.3,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     # Record time elapsed
#     finish_time = time.time()
#     elapsed_time = round(finish_time - start_time, 2)

#     # Parse the response
#     response = response[0]["generated_text"].split(request)[-1]

#     # Form output
#     results = {
#         "prompt": prompt,
#         "request": request,
#         "elapsed_time": elapsed_time,
#         "response": response
#     }
#     return results

# all_results = []
# for select_prompt in all_prompts:
#     single_result = timed_generation(prompt=select_prompt)
#     all_results.append(single_result)


# COMMAND ----------


# batch_coef = 1


# positive_reviews = [positive_review_example] * batch_coef

# formatted_requests = [
#     positive_prompt_4.format(review=pos_rev) 
#     for pos_rev in positive_reviews
# ]


# llm_responses = mpt_pipeline(
#     formatted_requests, 
#     max_new_tokens=150,
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.eos_token_id,
#     do_sample=True,
#     temperature=0.3,
#     # repetition_penalty=1.0,
#     # top_p=0.9,
#     # top_k=50,
#     # no_repeat_ngram_size=0,
#     batch_size=batch_coef
# )

# for response, f_request, r_request in zip(llm_responses, formatted_requests, positive_reviews):
#     _response = response[0]["generated_text"].split(f_request)[-1]
#     print("-"*10)
#     # print(f"Review:\n{r_request}")
#     print(f"\nMPT-7B:\n{_response}")
#     # print("-"*10)

# COMMAND ----------

# INSTRUCTION_KEY = "### Instruction:"
# RESPONSE_KEY = "### Response:"
# INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
# PROMPT_FOR_GENERATION_FORMAT = """{intro}
# {instruction_key}
# {instruction}
# {response_key}
# """.format(
#     intro=INTRO_BLURB,
#     instruction_key=INSTRUCTION_KEY,
#     instruction="{instruction}",
#     response_key=RESPONSE_KEY,
# )

# example = "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week? Explain before answering."
# fmt_ex = PROMPT_FOR_GENERATION_FORMAT.format(instruction=example)
# print(fmt_ex)

# COMMAND ----------

# # Build prompt template
# positive_prompt_1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ###Instruction:
# These are positive book reviews:

# {review}

# Identify three distinct and specific aspects that readers enjoyed about the book from these positive reviews, and provide a bullet point summary for each.
# ### Response:
# """

# positive_prompt_2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# From the text below composed of several book reviews, distill and provide three bullet points capturing what customers most appreciated about the book they reviewed.
# {review}
# ### Response:
# """

# positive_prompt_3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Using the text below, identify three aspects that readers liked about the book and provide a summary for each.
# {review}
# ### Response:
# """

# positive_prompt_4 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Input: {review}
# Identify three aspects that readers liked about the book and provide a summary for each.
# ### Response:
# """

# positive_prompt_5 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Given the following review text: '{review}', identify three aspects that readers appreciated about the book, and summarise them into three bulletpoints.
# ### Response:
# """

# positive_prompt_6 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Read the following combined reviews fro and provide a bullet-point summary capturing what the customers liked about this book:

# {review}

# Provide a three bullet-point summary capturing what customers liked about this book
# ### Response:
# """
