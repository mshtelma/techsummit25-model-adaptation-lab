# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
uc_target_catalog = "msh"
uc_target_schema = "test"
uc_volume_path = "/Volumes/msh/finreg/data"
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
docs_df = load_and_clean_data(uc_volume_path)
display(docs_df) # noqa
# COMMAND ----------

splitted_df = split(
    docs_df, hf_tokenizer_name="hf-internal-testing/llama-tokenizer", chunk_size=500
)
display(splitted_df) # noqa

# COMMAND ----------

splitted_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.splitted_documents")

# COMMAND ----------
