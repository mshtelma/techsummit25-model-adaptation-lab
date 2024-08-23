# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install databricks-genai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC

# COMMAND ----------
uc_target_catalog = "msh"
uc_target_schema = "test"
# COMMAND ----------


from databricks.model_training import foundation_model as fm

from finreganalytics.utils import setup_logging, get_dbutils, get_current_cluster_id

setup_logging()

supported_models = fm.get_models().to_pandas()["name"].to_list()
get_dbutils().widgets.combobox(
    "base_model", "meta-llama/Meta-Llama-3-8B-Instruct", supported_models, "base_model"
)
get_dbutils().widgets.text(
    "data_path", "/Volumes/main/finreg/training/ift/jsonl", "data_path"
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
data_path = get_dbutils().widgets.get("data_path")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")
custom_weights_path = get_dbutils().widgets.get("custom_weights_path")
if len(custom_weights_path) < 1:
    custom_weights_path = None
cluster_id = get_current_cluster_id()

# COMMAND ----------

run = fm.create(
    model=base_model,
    train_data_path=f"{uc_target_catalog}.{uc_target_schema}.qa_instruction_train",
    eval_data_path=f"{uc_target_catalog}.{uc_target_schema}.qa_instruction_val",
    register_to=f"{uc_target_catalog}.{uc_target_schema}.fin_reg_model",
    training_duration=training_duration,
    learning_rate=learning_rate,
    task_type="CHAT_COMPLETION",
    data_prep_cluster_id=cluster_id
)

# COMMAND ----------

display(fm.get_events(run))

# COMMAND ----------

run.name

# COMMAND ----------

display(fm.list())
