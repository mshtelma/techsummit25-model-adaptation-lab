# Databricks notebook source
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
uc_target_catalog = get_user_name()
uc_target_schema = get_user_name()
# COMMAND ----------


chunks_df = get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.splitted_documents")
chunks = chunks_df.toPandas()["text"].values.tolist()

# COMMAND ----------

llm_dbrx = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", temperature=0.9)
EVALUATION_QUESTION_GENERATION_PROMPT_TMPL = """\
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
QA_TEMPLATE_RAG = """
Context information is below.

---------------------
{context}
---------------------  

You are an expert in European Financial Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
If the question is not related to one of these topics, kindly decline to answer. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible.
Please do not repeat the answer and do not add any additional information. 

Question: {question}

Answer:
"""

# COMMAND ----------

qa_questions_df = build_instruction_eval_dataset(
    chunks[100:200],
    llm_dbrx,
    question_prompt_template_str=EVALUATION_QUESTION_GENERATION_PROMPT_TMPL,
    answer_prompt_template_str=QA_TEMPLATE_RAG,
    num_questions_per_chunk=2,
)
qa_questions_df = spark.createDataFrame(qa_questions_df)
display(qa_questions_df)  # noqa

# COMMAND ----------
qa_ift_df = prepare_ift_dataset(qa_questions_df, limit=-1)

ift_train_df, ift_val_df = qa_ift_df.randomSplit([0.9, 0.1])

ift_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train")
ift_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val")
