import functools
import io
import re
from typing import Iterator, List

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from pypdf import PdfReader
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, explode
from pyspark.sql.pandas.functions import pandas_udf
from transformers import AutoTokenizer

from finreganalytics.utils import get_spark


def load_and_clean_data(source_folder: str) -> DataFrame:
    """
    Loads PDFs from the specified folder

    :param source_folder: Folder with PDFs.
    :return: List of `Documents`
    """
    df = (
        get_spark()
        .read.format("binaryFile")
        .option("pathGlobFilter", "*.pdf")
        .load(source_folder)
        .repartition(20)
    )

    def clean(txt):
        txt = re.sub(r"\n", "", txt)
        return re.sub(r" ?\.", ".", txt)

    def parse_and_clean_one_pdf(b: bytes) -> str:
        pdf = io.BytesIO(b)
        reader = PdfReader(pdf)
        chunks = [page_content.extract_text() for page_content in reader.pages]
        return "\n".join([clean(text) for text in chunks])

    @pandas_udf("string")
    def parse_and_clean_pdfs_udf(
            batch_iter: Iterator[pd.Series],
    ) -> Iterator[pd.Series]:
        for series in batch_iter:
            yield series.apply(parse_and_clean_one_pdf)

    return df.select(col("path"), parse_and_clean_pdfs_udf("content").alias("text"))


def split(df: DataFrame, hf_tokenizer_name: str, chunk_size: int) -> DataFrame:
    """
    Splits documents into chunks of specified size
    :param docs: list of Documents to split
    :param hf_tokenizer_name: name of the tokenizer to use to count actual tokens
    :param chunk_size: size of chunk
    :return: list of chunks
    """

    def split(text: str, splitter: TextSplitter) -> List[str]:
        if len(text) > 750:
            return [
                doc.page_content
                for doc in splitter.split_documents([Document(page_content=text)])
            ]
        else:
            return [text]

    @pandas_udf("array<string>")
    def split_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(hf_tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
        )
        for series in batch_iter:
            yield series.apply(functools.partial(split, splitter=text_splitter))

    return df.select(col("path"), explode(split_udf("text")).alias("text"))
