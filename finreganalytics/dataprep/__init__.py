import os
import pathlib
import shutil

from pyspark.sql import DataFrame
from streaming.base.converters import dataframe_to_mds


def store_as_mds(sdf: DataFrame, path: str, overwrite: bool = True):
    if overwrite and os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)

    dataframe_to_mds(
        sdf.repartition(8),
        merge_index=True,
        mds_kwargs={"out": path, "columns": {col: "str" for col in sdf.columns}},
    )


def store_as_jsonl(sdf: DataFrame, filename: str, overwrite: bool = True):
    pathlibpath = pathlib.Path(filename)
    pathlibpath.parent.mkdir(exist_ok=True)

    sdf.toPandas().to_json(filename, orient="records", lines=True)
