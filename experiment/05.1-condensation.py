# Databricks notebook source


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- You can skip this line if no-UC
# MAGIC USE CATALOG mas;
# MAGIC
# MAGIC -- Sets the standard database to be used in this notebook
# MAGIC USE SCHEMA review_summarisation;
# MAGIC
# MAGIC

# COMMAND ----------

# Import the OS system to declare a ENV variable
import os

# Setting up the storage path (please edit this if you would like to store the data somewhere else)
main_storage_path = "/Volumes/mas/review_summarisation/model_store"

# Declaring as an Environment Variable 
os.environ["MAIN_STORAGE_PATH"] = main_storage_path

local_model_path = f"{main_storage_path}/mpt-7b-instruct/"
local_tokenizer_path = f"{main_storage_path}/mpt-7b-tokenizer/"

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

# Required tokenizer setting for batch inference
mpt_pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

condensation_text_1 = """Here are the three distinct aspects that readers enjoyed about the book:
1. The story is well written and engaging.
2. The characters are well developed and the reader is able to relate to them.
3. The book is a page turner and the reader is unable to put it down.
Here are the three distinct aspects that readers enjoyed about the book:
1. The book ties in the characters from the previous book, Feast of Crows.
2. The book provides more details about the characters that were not featured in the previous book.
3. The book has a cliffhanger that makes readers want to read the next book."""

condensation_text_2 = """The book is an honest account of Chris Kyle's life. The book is a tribute to Chris Kyle's bravery. The book is a historical memoir of Chris Kyle's life. The book provides a very candid and personal account of the life of a real American hero. The book provides a very detailed account of the author's experiences in combat and the toll it takes on all men and women involved. The book provides a very insightful look into the realities of modern warfare. The book was a great read. The book was well written and engaging. The book provided a great insight into the life of a true American hero. The book provides a glimpse into the heroic lives of Chris and Taya Kyle. The book highlights the sacrifices made by Chris and all military personnel and their families. The book brings the reality of war and its cost to the forefront. The book is a great read and easy to follow. The book provides a true story of a modern day war hero. The book is a great depiction of a true American hero. The book was well written and engaging. The book provided a unique perspective on war. The book was inspirational. The book helps people understand what service men and their families go through before, during, and after deployment. The book is well written and a must read for the current generation. The book is riveting and tells the story of courage and patriotism. The book is well written and provides a realistic perspective of the war in Iraq The book provides a detailed account of the life of a great man and a great woman The book provides a detailed account of the life of a great man and a great woman The book's vivid descriptions of war and combat. The book's authentic portrayal of the life of a soldier. The book's exploration of the impact of war on soldiers and their families. The book is very personal and well written. The book clearly illustrates the life of a true American hero. The book makes you think. The book provides a direct, revealing insight into what it takes to be a hero in the midst of constant hostile surroundings and weapons fire. The book gives illuminating insight into the upbringing and personal life of the author. The book is well done and gives you a picture of what combat is like. The book provides a brutally honest look into the psyche of a true hero. The book provides insight into the marriage and family of a true hero. The book helps the reader appreciate the many sacrifices of American service families. The book was well written and difficult to put down. The book was always exciting and didn't get bogged down with too much detail of the scenery. The book was a good read that was a little disjointed. The book provides a glimpse into the life of a real American hero. The book is very inspiring and motivating. The book is very informative. The dialog between Taya and Chris Kyle. The book's depiction of the war experience from the perspective of a Seal Sniper. The book's first person account of what actually went on during the war. The book provides a real view of the war in Iraq. The book paints a picture of a soldier's life when deployed. The book shows how war impacts the family at home. The book is an autobiography of Chris Kyle, a former Navy SEAL and the most lethal sniper in American history. The book is a first hand account of Chris Kyle's experiences in the Iraq War and the Afghanistan War. The book paints a very realistic portrayal of war that is not seen in the news. The book is about a true American hero. The book is about a true American hero who is a role model for people to look up to. The book is about a true American hero who is a role model for people to look up to and who is a true patriot. Chris Kyle defines the enemy as the barbarians who are the enemy. Chris Kyle is a true American hero. Chris Kyle and his wife experience conflicts in their struggle to deal with their love and their lives apart. Chris Kyle pursues what he wants in life and is dedicated to what is right. Chris Kyle is humble and never brags. Chris Kyle is a giant among men. The book provides a well written account of Chris Kyle's life and his time spent in Iraq as a Navy Seal. The book delves into the back story of Chris Kyle and the demons he faced. The book provides insight into the life of an American hero. The book clearly describes the patriotic efforts and feelings of an American soldier. The book also describes some of the prejudices and personal problems of the soldier. The book is beautifully written. The book is an honest and gripping account of a man torn between family and war. The book contains passages of the author's life in war and training. They are great histories and worth your time. The book is a tribute to the author's life and his love for the marines. The book is a very real down to earth story about a true American hero. The book gives a very insightful look into the mind of an American sniper. The book is a very honest portrayal of the author's thoughts and emotions. The book is written extremely authentically. The book does not glorify war. The book is a true American hero. Chris Kyle's book, American Sniper, is a memoir about his time as a Navy SEAL sniper in the Iraq War. The book is an autobiography that describes his experiences in combat, including the deaths of several of his friends. Kyle also discusses his personal life, including his marriage to Taya Kyle, and his struggles with post traumatic stress disorder (PTSD). The book is widely praised for its honest portrayal of war, and Kyle is praised for his bravery and dedication to his country. The book was about Chris Kyle, a Navy SEAL who served four tours in Iraq. He was known as the most lethal sniper in U.S. military history. The book was written by Chris Kyle and his wife Taya Kyle. The book was published in The book was about Chris Kyle's life, his time in the military, and his time after he left the military. The book was about Chris Kyle's life, his time in the military, and his time after he left the military. The book was a good read. The book was very insightful. The book made the reader feel like they were there with Chris Kyle."""


condensation_prompt_1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the summarized book reviews below and identify four distinct themes that encapsulate the aspects readers enjoyed about the book. Condense these into four succinct bullet points.

Summarized Reviews: {summarized_reviews}

### Response:
"""

condensation_prompt_2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the summarized book reviews below, looking for recurring and overlapping themes. Identify four distinct themes that encapsulate the aspects readers enjoyed about the book, and condense these into four succinct and non-redundant bullet points.

Summarized Reviews: {summarized_reviews}

### Response:
"""

condensation_prompt_3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the summarized book reviews below, and identify four overarching and non-redundant themes that encapsulate the aspects readers enjoyed about the book. Consider aspects like writing quality, character development, thematic continuity, and reader engagement. Condense these aspects into four succinct bullet points.

Summarized Reviews: {summarized_reviews}

### Response:
"""

condensation_prompt_4 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the summarized book reviews below, and identify four overarching and non-redundant themes that encapsulate the aspects readers enjoyed about the book. Each theme should cover a different aspect of the book, such as writing quality, character portrayal, thematic depth, and reader engagement. Be explicit about what aspect each theme is referring to and ensure that there is no redundancy among the themes. Condense these aspects into four succinct bullet points.

Summarized Reviews: {summarized_reviews}

### Response:
"""


condensation_prompt_5 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify five distinct aspects that readers enjoyed about the book. Return the result as five succinct bullet points.

Reviews: {summarized_reviews}

### Response:
"""

condensation_prompt_6 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the summarized book reviews below, and identify four overarching and non-redundant themes that encapsulate the aspects readers enjoyed about the book. Consider the following dimensions:
- Writing Style: How is the book's writing quality, language, and structure?
- Character Development: What do readers think about the portrayal of characters?
- Thematic Content: What themes, ideas, or subjects resonated with readers?
- Reader Engagement: How did the book engage, motivate, or inspire readers?
Condense these aspects into four succinct bullet points, each representing one of the above dimensions.

Summarized Reviews: {summarized_reviews}

### Response:
# """


condensation_prompt_x = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify the top five distinct aspects that readers enjoyed about the book. Return the result in five concise bullet points.

Reviews: {summarized_reviews}

### Response:
"""



# Generate response
llm_responses = mpt_pipeline(
    condensation_prompt_x.format(summarized_reviews=condensation_text_2),
    max_new_tokens=200,  # How long can the answer be ?
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.1,  # How creative the model can be ? (1 = max creativity)
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
    # no_repeat_ngram_size=0

)

print(llm_responses[0]["generated_text"].split("### Response:")[-1])

# COMMAND ----------


