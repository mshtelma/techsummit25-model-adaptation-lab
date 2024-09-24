# Databricks notebook source
# MAGIC %md 
# MAGIC # Model Adaptation Demo 
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model 
# MAGIC
# MAGIC In this demo we will generate synthetic question/answer data about Capital Requirements Regulation and after that will use this data to fine tune the Llama 3.0 8B model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic Data Generation
# MAGIC In this notebook we will use the chain of though (CoT) technique to create high quality questions and answers about Capital Requirements Regulation.
# MAGIC We will iterate over all the chunks we created in the first step and generate a question about the facts mentioned in the chunk. Then we will ask an LLM to answer this question using the provided chunk. 

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.chat_models.databricks import ChatDatabricks
from pyspark.sql.functions import rand

from finreganalytics.dataprep.ift_data_prep import (
    prepare_ift_dataset,
)
from finreganalytics.dataprep.qagen import build_instruction_eval_dataset
from finreganalytics.utils import get_spark, get_user_name, batchify

# COMMAND ----------

# MAGIC %md In the following cell, we will specify the target catalog and schema where we will store all the tables we create during this demo. 
# MAGIC If the catalog, schema or source data path is not defined, we will try to create a new catalog and schema and copy sample pdf files from the git repo. 

# COMMAND ----------

uc_target_catalog = "msh"
uc_target_schema = "test"

if (locals().get("uc_target_catalog") is None
        or locals().get("uc_target_schema") is None):
    uc_target_catalog = get_user_name()
    uc_target_schema = get_user_name()

# COMMAND ----------

# MAGIC %md 
# MAGIC In the following cell, we define the prompt to generate an initial question that corresponds to the chunk of text.

# COMMAND ----------

INITIAL_QUESTION_GENERATION_PROMPT_TMPL = """\
Context information is below.

---------------------
{context}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. 
Your task is to come up with a question for an upcoming  quiz/examination on Capital Requirements Regulation (CRR). 
The questions should be diverse in nature across the document. Restrict the questions to the
context information provided.
Please generate exactly one questions and no more.
Do not include any further information.

"""

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cell we will start to use CoT and ask an LLM to give reasons and thoughts behind the choice of this particlar question. We will also ask the LLM to provide some judgment as to whether this is a good question and also to provide some ideas for improvement. 

# COMMAND ----------

JUDGEMENT_QUESTION_GENERATION_PROMPT_TMPL = """\

Context information:
---------------------
{context}
---------------------

Question
---------------------
{question}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. 
Your task is to make a decision if the mentioned above question is good for an upcoming  quiz/examination on Capital Requirements Regulation (CRR). 
Please come up with some thoughts first about the question above, think if this is a good one for students. 
After that make a decision and explain why you think it is a good or bad one.

"""

# COMMAND ----------

# MAGIC %md
# MAGIC Now, as a final step in the question generation process, we will ask the LLM to improve the generated question using the thoughts and improvement ideas we generated in the previous cell.

# COMMAND ----------

IMPROVE_QUESTION_GENERATION_PROMPT_TMPL = """\
Context information:
---------------------
{context}
---------------------

Question
---------------------
{question}
---------------------

Judgement
---------------------
{judgement}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. 
Your task is to come up with a question for an upcoming  quiz/examination on Capital Requirements Regulation (CRR). 
The questions should be diverse in nature across the document. Restrict the questions to the
context information provided.
Above you have a question and a judgement about its quality.
Improve the question according to the judgement and rewrite it to address points indicated in the judgement. 
If the question is already perfect just output it without any modifications.
Do not include any further information and do not write if the question is good or bad or what you have modified.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we have generated our question in the previous cell, and in the next cell we will define 3 prompts to generate an answer using the same logic:
# MAGIC - Generate initial answer
# MAGIC - Generate thoughts and reasoning behind it, and then come up with some judgments and ideas for improvement.
# MAGIC - Use these ideas to improve the answer.

# COMMAND ----------

INITIAL_ANSWER_GENERATION_PROMPT_TMPL = """
Context information:
---------------------
{context}
---------------------

You are an expert in European Financial Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
If the question is not related to one of these topics, kindly decline to answer. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible.
Please do not repeat the answer and do not add any additional information. 
Please answer the question above using information given in context.

Question: {question}

Answer:
"""

JUDGEMENT_ANSWER_GENERATION_PROMPT_TMPL = """
Context information:
---------------------
{context}
---------------------

Question
---------------------
{question}
---------------------

Answer
---------------------
{answer}
---------------------

You are an expert in European Financial Regulation and Capital Requirements Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
Your task is to make a decision if the mentioned above answer fully and correctly answers the question mentioned above. 
Please come up with some thoughts first about the answer above, think if this is a correct and full answer. 
After that make a decision and explain why you think it is a good or bad one.

Question: {question}

Answer:
"""

IMPROVE_ANSWER_GENERATION_PROMPT_TMPL = """
Context information:
---------------------
{context}
---------------------

Question
---------------------
{question}
---------------------

Answer
---------------------
{answer}
---------------------

Judgement
---------------------
{judgement}
---------------------

You are an expert in European Financial Regulation and Capital Requirements Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
Your task is to improve the mentioned above answer using the provided above judgement and rewrite it to address points indicated in the judgement. 
If the answer is already perfect just output it without any modifications.
Do not include any further information and do not write if the question is good or bad or what you have modified.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will use Llama 3.1 70B to run all these prompts for each chunk. 
# MAGIC We will pass them to the `build_instruction_eval_dataset` function which will iterate over the chunks, build the final prompts and send them to Llama 3.1 70B.
# MAGIC In the next cell, we will generate just two questions to validate our approach

# COMMAND ----------

chunks_df = get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.splitted_documents").orderBy(rand())
chunks = chunks_df.toPandas()["text"].values.tolist()[:100]

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", temperature=0.9)

qa_questions_df = build_instruction_eval_dataset(
    chunks[:2],
    llm,
    initial_question_prompt_template_str=INITIAL_QUESTION_GENERATION_PROMPT_TMPL,
    judgement_question_prompt_template_str=JUDGEMENT_QUESTION_GENERATION_PROMPT_TMPL,
    improve_question_prompt_template_str=IMPROVE_QUESTION_GENERATION_PROMPT_TMPL,
    initial_answer_prompt_template_str=INITIAL_ANSWER_GENERATION_PROMPT_TMPL,
    judgement_answer_prompt_template_str=JUDGEMENT_ANSWER_GENERATION_PROMPT_TMPL,
    improve_answer_prompt_template_str=IMPROVE_ANSWER_GENERATION_PROMPT_TMPL,
)
display(qa_questions_df)  # noqa

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have validated the entire approach, we can run it iteratively over the entire dataset. We will generate multiple questions for each chunk and iterate over the entire dataset in chunks of 200 chunks. We will store the generated questions and answers for each chunk independently.

# COMMAND ----------

number_of_questions = 2
chunk_length = 200

for i in range(number_of_questions):
    print(f"Iteration: {i}\n")
    for current_chunk in batchify(chunks, chunk_length):
        print(f"Chunk length: {len(current_chunk)}\n")
        qa_questions_df = build_instruction_eval_dataset(
            current_chunk,
            llm,
            initial_question_prompt_template_str=INITIAL_QUESTION_GENERATION_PROMPT_TMPL,
            judgement_question_prompt_template_str=JUDGEMENT_QUESTION_GENERATION_PROMPT_TMPL,
            improve_question_prompt_template_str=IMPROVE_QUESTION_GENERATION_PROMPT_TMPL,
            initial_answer_prompt_template_str=INITIAL_ANSWER_GENERATION_PROMPT_TMPL,
            judgement_answer_prompt_template_str=JUDGEMENT_ANSWER_GENERATION_PROMPT_TMPL,
            improve_answer_prompt_template_str=IMPROVE_ANSWER_GENERATION_PROMPT_TMPL,
        )
        (
            get_spark()
            .createDataFrame(qa_questions_df)
            .write
            .mode("append")
            .saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset")
        )

# COMMAND ----------

display(get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset"))  # noqa

# COMMAND ----------

# MAGIC %md Now we should have all questions and answers ready and we can transform them to the instruction dataset formatted as chat completions. We will use `prepare_ift_dataset`

# COMMAND ----------

qa_train_df, qa_val_df = get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset").orderBy(
    rand()).randomSplit([0.9, 0.1])
qa_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset_train")
qa_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset_val")

qa_ift_train_df = prepare_ift_dataset(qa_train_df, limit=-1)
qa_ift_val_df = prepare_ift_dataset(qa_val_df, limit=-1)

qa_ift_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train")
qa_ift_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val")

# COMMAND ----------

display(get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val"))  # noqa
