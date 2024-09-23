# Databricks notebook source
# MAGIC %md
# MAGIC # Model Adaptation Demo
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model
# MAGIC
# MAGIC In this demo we will generate synthetic question/answer data about Capital Requirements Regulation and after that will use this data to dine tune the Llama 3.0 8B model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-Tuning
# MAGIC In this notebook we will fine-tune Llama 3.0 8B model on the generated synthetic dataset

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install databricks-genai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Let's define the parameters we will use to start fine-tuning

# COMMAND ----------

from databricks.model_training import foundation_model as fm

from finreganalytics.utils import setup_logging, get_dbutils, get_current_cluster_id, get_user_name

setup_logging()

uc_target_catalog = get_user_name()
uc_target_schema = get_user_name()

supported_models = fm.get_models().to_pandas()["name"].to_list()
get_dbutils().widgets.combobox(
    "base_model", "meta-llama/Meta-Llama-3-8B-Instruct", supported_models, "base_model"
)

get_dbutils().widgets.text("training_duration", "1ep", "training_duration")
get_dbutils().widgets.text("learning_rate", "1e-6", "learning_rate")
get_dbutils().widgets.text(
    "custom_weights_path",
    "",
    "custom_weights_path",
)

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")
custom_weights_path = get_dbutils().widgets.get("custom_weights_path")
if len(custom_weights_path) < 1:
    custom_weights_path = None
cluster_id = get_current_cluster_id()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can kick off fine-tuning run using Databricks Training API

# COMMAND ----------

run = fm.create(
    model=base_model,
    train_data_path=f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train",
    eval_data_path=f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val",
    register_to=f"{uc_target_catalog}.{uc_target_schema}.fin_reg_model",
    training_duration=training_duration,
    learning_rate=learning_rate,
    task_type="CHAT_COMPLETION",
    data_prep_cluster_id=cluster_id,
    eval_prompts=[
        "Which financial instruments can be used as CET1 instruments according to the CRR?",
        "What items are required to be deducted from CET1 capital under the CRR?",
    ]
)

# COMMAND ----------

# MAGIC %md Let's examine the run object now!

# COMMAND ----------

run

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can look into lifecylce of our Run

# COMMAND ----------

display(fm.get_events(run))

# COMMAND ----------

# MAGIC %md
# MAGIC We can also list all run we have started using Training API

# COMMAND ----------

display(fm.list())

# COMMAND ----------

# MAGIC %md Let's stop all runs now since we do not want to consume a lot of GPU resources!

# COMMAND ----------

for run in fm.list():
    fm.cancel(run.name)

# COMMAND ----------

# MAGIC %md By now all the Runs must be in "STOPPED" or "TERMINATING" state

# COMMAND ----------

display(fm.list())

# COMMAND ----------
