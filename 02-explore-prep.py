# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation
# MAGIC
# MAGIC Given that we have our raw dataframes ready, we will now use this notebook to do some exploration and pre-preocessing work to prepare our datasets.
# MAGIC
# MAGIC What we are aiming for here is to bring the data into a format which we can use with an LLM so that it is in a "ready to be summarised state". We will try to achieve this by first sampling reviews in a logical way to be able to capture the good and the bad aspects of the books we have got, and then build a text that contains multiple reviews to summarised.
# MAGIC
# MAGIC Our ultimate goal is to build tis summariser pipeline so we can have a **faster time to action** after we receive reviews, which can be considered as feedback on our products. Therefore, we need to make sure that we capture the most we can from the reviews, which means that we need to pay attention to both good and bad reviews, and prepare our data so that a good amount of both sides make it through. This is quite important from a sampling perspective, because chances are **most products recieve more positive reviews than negative** (hinting at an inbalanced dataset), so if we straight up implement a generic sampler, we might loose a good proportion of the negative reviews recieved by the products.
# MAGIC
# MAGIC We also need to ensure that the piece of text we are going to send to the LLM doesn't contain too much text, which has to do with the **context lenghts** - most of the LLMs' performance begin to degrade with longer context lengths both from an quality perspective (how good is the summary?) and performance perspective (how fast can it run?). So, we need to slice and dice our reviews in a respective way to create sensible batches of reviews.
# MAGIC
# MAGIC Lets begin!
# MAGIC
# MAGIC
# MAGIC **Setup Used:**
# MAGIC
# MAGIC - Runtime: 13.2 ML
# MAGIC - Cluster:
# MAGIC   - Machine: 16 CPU + 64 GB RAM (For Driver & Worker)
# MAGIC   - 2-8 Worker Auto Scaling

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initial Setup
# MAGIC
# MAGIC Here we will begin by setting up the data standards and reading the `raw_book_reviews_df` we created in the last notebook.

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG mas;
# MAGIC -- You can skip this line if no-UC
# MAGIC USE SCHEMA review_summarisation;
# MAGIC -- Sets the standard database to be used in this notebook

# COMMAND ----------

# Read the table
raw_book_reviews_df = spark.read.table("raw_book_reviews")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Column Selection
# MAGIC The dataframe has many columns, but we don't really need all of them for this specific task. Some might be better suited for other algorithms such as product recommenders, so lets pick and choose what might be useful for us.

# COMMAND ----------

# To see what we have, lets quickly display the data
display(raw_book_reviews_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC There are some columns like `tech1` or `fit` which we can remove from the dataframe as well as `also_buy`. Rather than picking the ones that we need to remove, lets focus on the ones we need which might make things easier

# COMMAND ----------

# The columns we want
focus_columns = [
    # Book Attributes
    "asin",
    "title",
    "brand AS author",
    # "price",
    "main_cat AS main_category",
    # Review Attributes
    "reviewerID AS reviewer_id",
    "reviewerName AS reviewer_name",
    "unixReviewTime AS unix_review_time",
    "overall AS star_rating",
    "summary AS review_summary",
    "reviewText AS review_text",
    "verified",
]

book_reviews_df = raw_book_reviews_df.selectExpr(focus_columns)

display(book_reviews_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### De-Duplicate
# MAGIC
# MAGIC We had some issues with duplicating records on the previous notebook, lets see if we can resolve that here.

# COMMAND ----------

print(f"Current row count: {book_reviews_df.count()}")

# COMMAND ----------

from pyspark.sql import functions as SF
# Creating a duplicated records dataframe
# Reasoning here: There cannot be a reviewer that reviewed the same book more than once at the same time
# Therefore, see if there are duplicated reviews
display(
    book_reviews_df
    .groupBy("asin", "reviewer_id", "unix_review_time").count()
    .filter(SF.col("count") > 1)
    .orderBy(SF.col("count").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like we do have some duplicates, lets see that they look like. There is also a clear pattern here, some specific books have problems with reviews.

# COMMAND ----------

display(
    book_reviews_df
    .filter(SF.col("asin") == "B001MQA3DU")
    .filter(SF.col("reviewer_id") == "A3EBCNHNQIP2Z3")
    .filter(SF.col("unix_review_time") == "1453248000")
)

# COMMAND ----------

# MAGIC %md
# MAGIC It seems like specifically this book, The Old Man and the Sea by Ernest Hemmingway, which is a great story about an unlucky fisherman, Santiago, had extra problems with duplicates. This is probably a source data problem. Lets check out a different book to see how it is

# COMMAND ----------

display(
    book_reviews_df
    .filter(SF.col("asin") == "B001MQA3DU")
    .filter(SF.col("reviewer_id") == "A3EBCNHNQIP2Z3")
    .filter(SF.col("unix_review_time") == "1454457600")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Same problem persists with this book too, lets deduplicate the dataframe by the asin, reviewer_id, and unix_review_time columns and see what we are left with

# COMMAND ----------

# Deduplicate by asin, reviewer_id, unix_review_time
book_reviews_df = book_reviews_df.dropDuplicates(["asin", "reviewer_id", "unix_review_time"])

# Check to see if the problem is fixed
display(
    book_reviews_df
    .filter(SF.col("asin") == "B001MQA3DU")
    .filter(SF.col("reviewer_id") == "A3EBCNHNQIP2Z3")
    .filter(SF.col("unix_review_time") == "1453248000")
)

# COMMAND ----------

# MAGIC %md
# MAGIC What about same reviewer reviewing the same item multiple times ? Is this possible or should we allow it ? Lets have a look..

# COMMAND ----------

# Check to see if the same reviewer revied the same book multiple times
display(
    book_reviews_df
    .groupBy("asin", "reviewer_id").count()
    .filter(SF.col("count") > 1)
    .orderBy(SF.col("count").desc())
)

# COMMAND ----------

display(
    book_reviews_df
    .filter(SF.col("asin") == "1607105551")
    .filter(SF.col("reviewer_id") == "A1D2C0WDCSHUWZ")
)

# COMMAND ----------

display(
    book_reviews_df
    .groupBy("asin", "reviewer_id").count()
    .filter(SF.col("count") > 1)
    .orderBy(SF.col("count").desc())
    .groupby().sum("count")
    # .count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC These reviews where the same reviewer reviewed the same product multiple times don't look natural to me. It might be the work of bots or something else. A quick count of these shows that there are 189k of these reviews. We can safely drop them, given that they make a small fraction of our total reviews (we have 51 million reviews)

# COMMAND ----------

print(f"Current count: {book_reviews_df.count()}")

de_dupe_df = (
    book_reviews_df
    .groupBy("asin", "reviewer_id")
    .count()
    .filter(SF.col("count") > 1)
    .orderBy(SF.col("count").desc())
    .select("asin", "reviewer_id")
)

book_reviews_df = book_reviews_df.join(de_dupe_df, on=["asin", "reviewer_id"], how="leftanti")
print(f"After count: {book_reviews_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Date Transformations
# MAGIC Given the UNIX time, we want to extract things like date and time, year month, and year week so we can slice and dice our data properly with different time ranges

# COMMAND ----------

from pyspark.sql import functions as SF

book_reviews_df = (
    book_reviews_df
    .withColumn("review_date", SF.from_unixtime("unix_review_time").cast("date"))
    .withColumn("week_start", SF.expr("date_sub(review_date, (dayofweek(review_date) - 2) % 7)"))
    .withColumn("month_start", SF.expr("trunc(review_date, 'MM')"))
    .withColumn("year_start", SF.expr("trunc(review_date, 'YYYY')"))
)

display(book_reviews_df)

# COMMAND ----------

display(
    book_reviews_df
    .groupBy("week_start", "year_start")
    .agg(SF.count("reviewer_id"))
    .orderBy(SF.col("week_start"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC What we can tell from above is that majority of the reviews came after 2012 January. Lets take a closer look there

# COMMAND ----------

display(
    book_reviews_df
    .withColumn("time_split", SF.col("review_date") >= "2012-01-01")
    .groupBy("time_split")
    .agg(SF.count("reviewer_id"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verified vs Non-Verified Purchases
# MAGIC Lets take a look at verified vs non verified reviewers. The difference is - if a review is verified, it means that the review came from someone who actually purchased the book. Non-verified reviews may include false reviews or inputs from bots who try to spam reviews for multiple reasons. Ideally, we want to keep our data as high quality as possible. So lets see what our distribution looks like over there 

# COMMAND ----------

from pyspark.sql import functions as SF

display(
    book_reviews_df
    .groupBy("week_start", "verified")
    .agg(SF.count("reviewer_id").alias("review_count"))
    .orderBy("week_start", "verified")
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The percentage of the reviews verified increases as time passes, which we can see from the graph above. It shows that in 2013, almost 80% of the recived reviews were from verified purchaers as opposed to 2005 where only 10% of the reviews were verified.
# MAGIC
# MAGIC Lets take a look at the percentage of reviews that happened after 2012 Jan which were verified

# COMMAND ----------

display(
    book_reviews_df
    .filter(SF.col("review_date") >= "2012-01-01")
    .groupBy("verified")
    .agg(SF.count("reviewer_id").alias("review_count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Time & Verification Filters
# MAGIC This means that if we take the reviews that happened after 2012, and only include the ones that were verified, we will still have about 32.2 million reviews to work with. This is still a good number, and it can help us with increasing the data quality since we want to make sure that the reviews came from people who actually purchased the products, and using recent data is never a bad idea. Lets apply our filters accordingly:

# COMMAND ----------

# Import pyspark sql functions
from pyspark.sql import functions as SF

# Get count before
print(f"Before count: {book_reviews_df.count()}")

# Apply filters
book_reviews_df = (
    book_reviews_df
    .filter(SF.col("review_date") >= "2012-01-01") 
    .filter(SF.col("verified") == True)
)

# Get count after
print(f"After count: {book_reviews_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Review Text & Headline Cleaning
# MAGIC
# MAGIC Some of the reviews and the headlines contain unexpected characters like HTML code or links. We want to get rid of those to make sure only relevant information stays on in the text

# COMMAND ----------

# Taking a look at potential problems
display(
    book_reviews_df
    .filter(SF.col("review_text").rlike("<a|</a>|href=|hook="))
    .select("review_text")
)

# COMMAND ----------

# External Imports
from pyspark.sql import functions as SF
import re

# Build Regex for cleaning text
remove_regex = re.compile(r"(&[#0-9]+;|<[^>]+>|\[\[[^\]]+\]\]|[\r\n]+)")
split_regex = re.compile(r"([?!.]\s+)")

@SF.udf("string")
def clean_text(text):
    if not text:
        return ""
    text = remove_regex.sub(" ", text.strip()).strip()
    cleaned = ""
    for fragment in split_regex.split(text):
        cleaned += fragment
    return cleaned.strip()

book_reviews_df = (
    book_reviews_df
    .withColumn("review_text", clean_text(SF.col("review_text")))
    .withColumn("review_summary", clean_text(SF.col("review_summary")))
)



# COMMAND ----------

# Running the same to see if we are good to go
book_reviews_df = (
    book_reviews_df
    .filter(~SF.col("review_text").rlike("<a|</a>|href=|hook="))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like there is still a few thats coming though, however we can try to deal with those in the next section

# COMMAND ----------

# MAGIC %md
# MAGIC #### Token Count Calculation
# MAGIC A term which we are going to see a lot in the upcoming notebooks is Tokens. You can think of tokens like words, but they are a bit different. Large Language Models first transform text in to tokens, which can be interpretted as the way that they read text. These tokens are then encoded in to numbers, which are the mathmetical representations of the text pieces we see here. In thoery, a large language model actually never sees a word! It only knows of numbers. We can think of these tokenizers as their language translator. 
# MAGIC
# MAGIC Token count becomes an important aspect in this sense, because each LLM has a pre-specified context length, which is bound by token count. For example, if you hear that an LLM has a context length of 2k, that would mean that the longest text it can process can have at most 2048 tokens or so. If you try to feed it more than that, it will error. 
# MAGIC
# MAGIC Therefore, we need to be careful of how many tokens we generate.. Of course, each tokenizer generates a different number of tokens, but we can get to an approximation using the TikToken Library. Lets see how that can be done:
# MAGIC
# MAGIC A simple math to go from word count to token count is:
# MAGIC
# MAGIC `n_tokens = n_words * 1.2`
# MAGIC
# MAGIC This can definitely change, and greatly depends on the tokenizer used. However as a ballpark figure we can use this.
# MAGIC

# COMMAND ----------

# External Imports
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as SF
import tiktoken

# Build token counter UDF
@SF.udf(IntegerType())
def calculate_n_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(string))
    except TypeError:
        # We implement this part to be able to deal with text that cannot be encoded
        num_tokens = -1
    return num_tokens

# Apply Function
book_reviews_df = book_reviews_df.withColumn("review_n_tokens", calculate_n_tokens("review_text"))

# Display
display(book_reviews_df.select("title", "review_date", "review_text", "review_n_tokens").limit(5))

# COMMAND ----------

# Check out the negative values (cannot be encoded)
display(
    book_reviews_df
    .filter(SF.col("review_n_tokens") == -1)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Even though there were some small examples left which contained some unexpected characters, there wasn't any text which our encoder couldn't deal with, which is great news

# COMMAND ----------

# MAGIC %md
# MAGIC #### Review Length
# MAGIC
# MAGIC Now that we have our verified and recent reviews and counted the number of token in each, lets take a look at the review lengths. From what we can tell by some manual inspection, there are some reviews which are quite short (less than 3-4 words long.) Lets check what those look like.

# COMMAND ----------

display(book_reviews_df)

# COMMAND ----------

# MAGIC %md
# MAGIC From what we can tell from this histogram, most of the reviews are less than 200 tokens long. But what does that look like reality ? Also, there is a long tail which we need to get rid of. 

# COMMAND ----------

# What does 500 tokens look like ?
import textwrap

print(
    textwrap.fill(
            book_reviews_df
            .filter(SF.col("review_n_tokens").between(490, 510))
            .limit(1)
            .select("review_text")
            .collect()[0][0],
        width=80,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC 500 tokens looks like a lot, we will need to make a decision about what do with these. We will have two options:
# MAGIC
# MAGIC 1. Filter out reviews longer than x amount of tokens
# MAGIC 2. Cut the reviews short, meaning that take only the first x amount of tokens.
# MAGIC
# MAGIC Both will come with some information loss. However, 2nd option might be better because it might still tell us something with the risk of context loss (now knowing about the entire text).
# MAGIC
# MAGIC Before we make a decision, lets see if there is a relationship between token length and star rating. My guess is that a disappointed reviewer might leave a longer review/complaint..

# COMMAND ----------

# MAGIC %md
# MAGIC #### Star Rating Distribution
# MAGIC Each review comes with a star rating which is a scale out of 5, 5 being the highest score (highest satisfaction) and 1 being the lower score (completely dissatisfied). Lets take a look at how this distribution is.

# COMMAND ----------

from pyspark.sql import functions as SF

display(
    book_reviews_df
    .groupBy("star_rating")
    .agg(SF.count("reviewer_id"))
)


# COMMAND ----------

# MAGIC %md
# MAGIC From what we can tell from this pie chart, there is a big imbalance towards 5s & 4s. Combined, they make up for 85% of the reviews. We have to be careful about this imbalance while sampling.
# MAGIC
# MAGIC Also, there are two reviews with 0 rating ?.. Have to investigate

# COMMAND ----------

display(
    book_reviews_df
    .filter(SF.col("star_rating") == 0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC We'll just assume they are not very happy.. 
# MAGIC
# MAGIC Lets see how the ratings compare against token count 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Star Rating vs Token Length

# COMMAND ----------

from pyspark.sql import functions as SF

display(
    book_reviews_df
    .filter(SF.col("star_rating") != 0)
    .groupBy("star_rating")
    .agg(
        SF.min(SF.col("review_n_tokens")).alias("min_n"),
        SF.median(SF.col("review_n_tokens")).alias("median_n"),
        SF.avg(SF.col("review_n_tokens")).alias("avg_n"),
        SF.max(SF.col("review_n_tokens")).alias("max_n"),
        SF.stddev(SF.col("review_n_tokens")).alias("std_n"),
    )
    .orderBy("star_rating")
)

# COMMAND ----------

# MAGIC %md
# MAGIC It does look like reviews get shorter by their avg and median token count, and standard deviation gets smaller as we move up from 1 to 5 ratings. Our assumption was correct, with the rating 2 being an outlier.
# MAGIC
# MAGIC We can also tell that the average review is shorter than 70 tokens for all rating groups and the standard deviation is at 136, which means that 70 + 136 = 206 token length will be long enough to capture most of the reviews.
# MAGIC
# MAGIC Lets see how many reviews have a token count greater than 200 per star rating

# COMMAND ----------

from pyspark.sql import functions as SF

display(
    book_reviews_df
    .filter(SF.col("star_rating") != 0)
    .withColumn("is_long_token", SF.col("review_n_tokens") > 210)
    .groupBy("star_rating", "is_long_token")
    .agg(SF.count("reviewer_id"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like more or less 5% of the reviews are too long for each rating. It's good to know that not much of the data will change when we deal with this. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Review Length Shortening
# MAGIC
# MAGIC Lets go ahead and shorten the reviews rather than filtering them out, and set the max_n_tokens to be 200.

# COMMAND ----------

# External Imports
import tiktoken
from pyspark.sql import functions as SF

# Function to count tokens using tiktoken
@SF.udf("string")
def truncate_text(text):
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = list(encoder.encode(text))
    if len(tokens) > 200:
        text = encoder.decode(tokens[:200])
    return text

book_reviews_df = (
    book_reviews_df
    .withColumn("review_text", truncate_text(SF.col("review_text"))) # Truncate
    .withColumn("review_n_tokens", calculate_n_tokens("review_text")) # Recount
)

# COMMAND ----------

display(book_reviews_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We now have a much better distribution of tokens

# COMMAND ----------

# MAGIC %md
# MAGIC #### Star Rating Classes
# MAGIC
# MAGIC Given that we are dealing with an inbalanced dataset, we need to deal with the way we process our reviews carefully. If we just sample and take a bunch of reviews, and then try to understand how a product can be improved, we might run into some problems because our sample dataset will be overwhelmed with positive reviews. Therefore, what we can do is create two classes - high_score and low_score. 
# MAGIC
# MAGIC This can assume that a customer who rates a product with 4 or 5 stars is giving it a high score, meaning that they are most probably sattisfied, and a customer that gives anything lower than that is giving it a low score, meaning that they are not satisfied with their purchase. 
# MAGIC
# MAGIC Then, we can use the high scores to understand what the customers are happy with, and the low scores to understand what can be improved, or what the customers disliked. 
# MAGIC
# MAGIC Building these classes are quite easy:

# COMMAND ----------

from pyspark.sql import functions as SF

book_reviews_df = (
    book_reviews_df
    .withColumn("star_rating_class", SF.when(SF.col("star_rating") > 3, "high").otherwise("low"))
)

display(book_reviews_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extract Author Name
# MAGIC
# MAGIC Not sure if it will be needed, but we can extract the Author name. In some cases, author's name is displayed as `Visit Amazon's William Shakespeare Page` probably hinting towards a link. We can easily change this so we extract just the name

# COMMAND ----------

from pyspark.sql import functions as SF

@SF.udf("string")
def extract_author_name(text):
    if text.startswith("Visit Amazon"):
        text = text.split("Amazon's")[-1]
        text = text.rsplit("Page", 1)[0]
        text = text.strip()
    return text

book_reviews_df = (
    book_reviews_df
    .withColumn("author", extract_author_name(SF.col("author")))
)

display(book_reviews_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build Books Metadata
# MAGIC
# MAGIC Similar to the metadata table we processed in the last notebook, we can go ahead and build a new metadata table, but this time include metrics like total reviews, first review data, total tokens, average score, etc..

# COMMAND ----------

from pyspark.sql import functions as SF

books_df = (
    book_reviews_df
    .repartition(256)
    .groupBy("asin", "title", "author")
    .agg(
        SF.min("review_date").alias("first_review_date"),
        SF.max("review_date").alias("last_review_date"),
        SF.count("*").alias("review_count"),
        SF.countDistinct("reviewer_id").alias("n_unique_reviewers"),
        SF.round(SF.avg("star_rating"), 2).alias("avg_rating"),
        SF.sum("review_n_tokens").alias("total_tokens"),
        SF.round(SF.avg("review_n_tokens"), 2).alias("avg_tokens")
    )
    .orderBy(SF.col("review_count").desc())
)

display(books_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   raw_books
# MAGIC WHERE
# MAGIC   asin = "0349403759"

# COMMAND ----------

display(
    books_df
    .limit(1000)
    .groupBy().sum("review_count"))

# COMMAND ----------


