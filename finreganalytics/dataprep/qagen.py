import json
import re
from operator import itemgetter
from typing import Union, List

import pandas as pd
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)


def parse(s: str, llm: BaseLanguageModel) -> Union[List[str], None]:
    """
    Tries parsing string into a json array
    :param s: string to parse
    :param llm: LLM to fix syntax
    :return: parsed list of questions
    """
    try:
        arr = json.loads(extract_json_array(s))
        if arr:
            return [r.strip() for r in arr]
        else:
            return None
    except Exception as e:
        return None
        # if llm:
        #     return improve_json(extract_json_array(s), llm)
        # else:
        #     raise e


def extract_json_array(s: str) -> str:
    """
    Strips json array from the surrounding text
    :param s: string with json
    :return: string which contains just an array
    """
    groups = re.search(r"\[.*]", s, re.DOTALL)
    if groups:
        return groups.group()
    else:
        return s


def build_instruction_eval_dataset(
        chunks: List[str],
        llm: BaseLanguageModel,
        question_prompt_template_str: str,
        answer_prompt_template_str: str,
        num_questions_per_chunk: int = 2,
) -> pd.DataFrame:
    """
    Generates an evaluation dataset containing Question, Answer and Context records using supplied LLM
    :param chunks: chunks to generate fact questions for
    :param llm: LLM to use for the question/answer generation
    :param question_prompt_template_str: Prompt template for the question generation
    :param answer_prompt_template_str: Prompt template to answer the generated question
    :param num_questions_per_chunk: Number of questions to generate per chunk
    :return: Pandas DataFrame containing generated questions
    """
    question_prompt = PromptTemplate(
        template=question_prompt_template_str,
        input_variables=["context", "num_questions_per_chunk"],
    )
    answer_prompt = PromptTemplate(
        template=answer_prompt_template_str,
        input_variables=["question", "context"],
    )
    questions_chain = RunnableParallel(
        context=RunnablePassthrough(),
        num_questions_per_chunk=RunnableLambda(lambda x: num_questions_per_chunk),
    ) | RunnableParallel(
        context=itemgetter("context"),
        question=question_prompt | llm | StrOutputParser(),
    ).with_retry(
        stop_after_attempt=10, wait_exponential_jitter=False
    )

    questions_results = questions_chain.batch(chunks, config={"max_concurrency": 4})
    questions_df = pd.DataFrame(
        [
            {
                "context": entry["context"].strip(),
                "question": parse(entry["question"], llm=llm),
            }
            for entry in questions_results
        ]
    )
    questions_df = questions_df.explode("question")
    questions_dict_list = questions_df.to_dict(orient="records")
    answers_chain = RunnableParallel(
        context=itemgetter("context"),
        question=itemgetter("question"),
        answer=answer_prompt | llm | StrOutputParser(),
    ).with_retry(stop_after_attempt=10, wait_exponential_jitter=False)
    answers_results = answers_chain.batch(
        questions_dict_list, config={"max_concurrency": 4}
    )
    res_df = pd.DataFrame(answers_results).dropna()
    return res_df
