# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()
# COMMAND ----------

from langchain_community.chat_models.databricks import ChatDatabricks

from finreganalytics.dataprep.ift_data_prep import (
    prepare_ift_dataset,
)
from finreganalytics.dataprep.qagen import build_instruction_eval_dataset
from finreganalytics.utils import get_spark, get_user_name, batchify

# COMMAND ----------
uc_target_catalog = "msh"
uc_target_schema = "finreg2"
# COMMAND ----------


chunks_df = get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.splitted_documents")
chunks = chunks_df.toPandas()["text"].values.tolist()

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
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", temperature=0.9)

qa_questions_df = build_instruction_eval_dataset(
    chunks[300:305],
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
number_of_questions = 20

(
    get_spark()
    .createDataFrame(qa_questions_df)
    .write
    .mode("append")
    .saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset")
)

for i in range(number_of_questions):
    for current_chunk in batchify(chunks, 500):
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
            .write()
            .mode("append")
            .saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset")
        )

# COMMAND ----------
display(get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset"))  # noqa
# COMMAND ----------
qa_ift_df = prepare_ift_dataset(get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset"), limit=-1)

ift_train_df, ift_val_df = qa_ift_df.randomSplit([0.9, 0.1])

ift_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train")
ift_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val")
