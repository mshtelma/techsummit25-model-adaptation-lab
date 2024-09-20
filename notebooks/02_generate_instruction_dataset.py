# Databricks notebook source
# MAGIC %md
# MAGIC # Model Adaptation Demo
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model
# MAGIC
# MAGIC In this demo we will generate synthetic question/answer data about Capital Requirements Regulation and after that will use this data to dine tune the Llama 3.0 8B model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic Data Generation
# MAGIC In this notebook we will use the CoT technique to create high quality questions and answers about Capital Requirements Regulation.
# MAGIC We will iterate over all the chunks we created in the first step and generate a question about the facts mentioned in the chunk and then ask an LLM to answer this question using the provided chunk.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.chat_models.databricks import ChatDatabricks

from finreganalytics.dataprep.ift_data_prep import (
    prepare_ift_dataset,
)
from finreganalytics.dataprep.qagen import build_instruction_eval_dataset
from finreganalytics.utils import get_spark, get_user_name

# COMMAND ----------

# MAGIC %md In the following cell we will specify the target catalog and schema where we will store all the tables we create during this demo.

# COMMAND ----------

uc_target_catalog = get_user_name()
uc_target_schema = get_user_name()

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC In the following cell, we will read all the splitted documents using Spark and transform them to a simple python list. 

# COMMAND ----------

chunks_df = get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.splitted_documents")
chunks = chunks_df.toPandas()["text"].values.tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cell, we define the prompt template to generate specified number of questions using provided chunk of text.

# COMMAND ----------

QUESTION_GENERATION_PROMPT_TMPL = """\
Context information is below.

---------------------
{context}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination on Capital Requirements Regulation (CRR). The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided.
Please generate exactly {num_questions_per_chunk} questions and no more. 
Do not include any further information.

Below is an example of a question.
Always format the output in JSON format as follows:

```json
[ 
"What problems addresses Capital Requirements Regulation?",
"What is Common Reporting Framework (COREP) ?" 
] 
``` """


# COMMAND ----------

# MAGIC %md
# MAGIC Please define the prompt template in the `ANSWER_TEMPLATE_RAG` variable that will ask an LLM to answer the question using the provided cointext. 
# MAGIC You can use the previous prompt template as an example. 
# MAGIC The prompt template must use two variables:
# MAGIC - `context` - for provided context (chunk of text)
# MAGIC - `question` - for the question an LLM must answer

# COMMAND ----------

ANSWER_TEMPLATE_RAG = """


"""

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can use the prompts we have defined in the previous cell to build question and answer dataset. 

# COMMAND ----------

llm_dbrx = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", temperature=0.9)

qa_questions_df = build_instruction_eval_dataset(
    chunks[100:200],
    llm_dbrx,
    question_prompt_template_str=QUESTION_GENERATION_PROMPT_TMPL,
    answer_prompt_template_str=ANSWER_TEMPLATE_RAG,
    num_questions_per_chunk=2,
)
qa_questions_df = spark.createDataFrame(qa_questions_df)
display(qa_questions_df)  # noqa

# COMMAND ----------

# MAGIC %md
# MAGIC After we have built the dataset containing questions and answers, it's time to transform it to the instructions using chat completion format.

# COMMAND ----------

qa_ift_df = prepare_ift_dataset(qa_questions_df, limit=-1)

ift_train_df, ift_val_df = qa_ift_df.randomSplit([0.9, 0.1])

ift_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train")
ift_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val")
