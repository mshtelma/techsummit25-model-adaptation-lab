from typing import List, Dict

from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

SYSTEM_INSTRUCTION = """You are a Regulatory Reporting Assistant.
Please answer the question as precise as possible using information in context.
If you do not know, just say I don't know. """


def format_chat_completion(
        context: str, question: str, answer: str
) -> Dict[str, List[Dict[str, str]]]:
    messages = []
    messages.append({"role": "system", "content": SYSTEM_INSTRUCTION})
    messages.append(
        {
            "role": "user",
            "content": f"""Context:\n {context}\n\n Please answer the user question using the given context:\n {question}""",
        }
    )
    messages.append({"role": "assistant", "content": answer})

    return messages


def transform_chat_udf(iterator):
    for df in iterator:
        df["messages"] = df.apply(
            lambda row: format_chat_completion(row["context"], row["question"], row["answer"]),
            axis=1,
        )
        df = df[["messages"]]
        yield df


def prepare_ift_dataset(
        spark_df: DataFrame = None,
        limit: int = -1,
) -> DataFrame:
    schema = StructType([
        StructField("messages", ArrayType(
            StructType([
                StructField("content", StringType(), nullable=True),
                StructField("role", StringType(), nullable=True)
            ])
        ))
    ])
    if limit > 0:
        spark_df = spark_df.limit(limit)
    transformed_sdf = spark_df.mapInPandas(transform_chat_udf, schema=schema)
    return transformed_sdf
