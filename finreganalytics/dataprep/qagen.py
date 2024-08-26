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
        initial_question_prompt_template_str: str,
        judgement_question_prompt_template_str: str,
        improve_question_prompt_template_str: str,
        initial_answer_prompt_template_str: str,
        judgement_answer_prompt_template_str: str,
        improve_answer_prompt_template_str: str,
        concurrency: int = 2,
        stop_after_attempt:int = 100
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
    initial_question_prompt = PromptTemplate(
        template=initial_question_prompt_template_str,
        input_variables=["context"],
    )
    judgement_question_prompt = PromptTemplate(
        template=judgement_question_prompt_template_str,
        input_variables=["context", "question"],
    )
    improve_question_prompt = PromptTemplate(
        template=improve_question_prompt_template_str,
        input_variables=["context", "question", "judgement"],
    )

    initial_answer_prompt = PromptTemplate(
        template=initial_answer_prompt_template_str,
        input_variables=["context", "question"],
    )
    judgement_answer_prompt = PromptTemplate(
        template=judgement_answer_prompt_template_str,
        input_variables=["context", "question", "answer"],
    )
    improve_answer_prompt = PromptTemplate(
        template=improve_answer_prompt_template_str,
        input_variables=["context", "question", "answer", "judgement"],
    )

    question_chain = (RunnableParallel(
        context=RunnablePassthrough(),
    ) | RunnableParallel(
        context=itemgetter("context"),
        question=initial_question_prompt | llm | StrOutputParser(),
    ) | RunnableParallel(
        context=itemgetter("context"),
        question=itemgetter("question"),
        judgement=judgement_question_prompt | llm | StrOutputParser(),
    ) | RunnableParallel(
        context=itemgetter("context"),
        question=itemgetter("question"),
        judgement=itemgetter("judgement"),
        final_question=improve_question_prompt | llm | StrOutputParser(),
    ).with_retry(
        stop_after_attempt=stop_after_attempt, wait_exponential_jitter=False
    ))

    questions_results = question_chain.batch(chunks, config={"max_concurrency": concurrency})
    questions_df = pd.DataFrame(
        [
            {
                "context": entry["context"].strip(),
                "initial_question": entry["question"],
                "question_judgement": entry["judgement"],
                "question": entry["final_question"],
            }
            for entry in questions_results
        ]
    )
    questions_df = questions_df.explode("question")
    questions_dict_list = questions_df.to_dict(orient="records")
    answer_chain = (RunnableParallel(
        context=itemgetter("context"),
        question=itemgetter("question"),
        initial_question=itemgetter("initial_question"),
        question_judgement=itemgetter("question_judgement"),
    ) | RunnableParallel(
        context=itemgetter("context"),
        question=itemgetter("question"),
        initial_question=itemgetter("initial_question"),
        question_judgement=itemgetter("question_judgement"),
        answer=initial_answer_prompt | llm | StrOutputParser(),
    ) | RunnableParallel(
        context=itemgetter("context"),
        question=itemgetter("question"),
        initial_question=itemgetter("initial_question"),
        question_judgement=itemgetter("question_judgement"),
        answer=itemgetter("answer"),
        judgement=judgement_answer_prompt | llm | StrOutputParser(),
    ) | RunnableParallel(
        context=itemgetter("context"),
        question=itemgetter("question"),
        initial_question=itemgetter("initial_question"),
        question_judgement=itemgetter("question_judgement"),
        answer=itemgetter("answer"),
        judgement=itemgetter("judgement"),
        final_answer=improve_answer_prompt | llm | StrOutputParser(),
    ).with_retry(
        stop_after_attempt=stop_after_attempt, wait_exponential_jitter=False
    ))
    answers_results = answer_chain.batch(
        questions_dict_list, config={"max_concurrency": concurrency}
    )
    res_df = pd.DataFrame([
        {
            "context": entry["context"].strip(),
            "initial_question": entry["initial_question"],
            "question_judgement": entry["question_judgement"],
            "question": entry["question"],
            "initial_answer": entry["answer"],
            "answer_judgement": entry["judgement"],
            "answer": entry["final_answer"],
        }
        for entry in answers_results
    ]).dropna()
    return res_df
