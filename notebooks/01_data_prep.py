# Databricks notebook source
# MAGIC %md 
# MAGIC # Model Adaptation Demo 
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model 
# MAGIC
# MAGIC In this demo we will generate synthetic question/answer data about Capital Requirements Regulation and after that will use this data to dine tune the Llama 3.0 8B model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC In the following cell we will specify the target catalog and schema where we will store all the tables we create during this demo. 

# COMMAND ----------

uc_target_catalog = "msh"
uc_target_schema = "test"
uc_volume_path = "/Volumes/msh/finreg/data"

# COMMAND ----------

# MAGIC %md
# MAGIC If the catalog, schema or source data path is not defined, we will try to create a new catalog and schema and copy sample pdf files from the git repo. 

# COMMAND ----------

import pathlib
import shutil

from databricks.sdk import WorkspaceClient

from finreganalytics.dataprep.dataloading import load_and_clean_data, split
from finreganalytics.utils import get_user_name, set_or_create_catalog_and_database

w = WorkspaceClient()

if (locals().get("uc_target_catalog") is None
        or locals().get("uc_target_schema") is None
        or locals().get("uc_volume_path") is None):
    uc_target_catalog = get_user_name()
    uc_target_schema = get_user_name()
    uc_volume_path = f"/Volumes/{uc_target_catalog}/{uc_target_schema}/data"
    set_or_create_catalog_and_database(uc_target_catalog, uc_target_schema)

    workspace_data_path = str((pathlib.Path.cwd() / ".." / "data").resolve())
    try:
        shutil.copytree(workspace_data_path, uc_volume_path, dirs_exist_ok=True)
    except Exception as e:
        print(e)
    w.dbutils.fs.ls(uc_volume_path)

# COMMAND ----------

# MAGIC %md Now we can ingest the pdf files and  parse their content

# COMMAND ----------

docs_df = load_and_clean_data(uc_volume_path)
display(docs_df) # noqa

# COMMAND ----------

# MAGIC %md After ingesting pdfs and transforming them tot he simple text, we will split the documents and store the chunks as a delta table

# COMMAND ----------

splitted_df = split(
    docs_df, hf_tokenizer_name="hf-internal-testing/llama-tokenizer", chunk_size=500
)
display(splitted_df) # noqa

# COMMAND ----------

# MAGIC %md Now let's store the chunks as a delta table

# COMMAND ----------

splitted_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.splitted_documents")
