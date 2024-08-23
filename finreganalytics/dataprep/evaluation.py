from typing import List, Dict

import mlflow
import pandas as pd


def run_chain_for_eval_data(chain, input_prompts: List[Dict[str, str]]) -> List[str]:
    """
    Generates answers for the list of questions using defined LCEL chain
    :param chain: chain to use for inference
    :param input_prompts: list of questions
    :return: list of resulting strings
    """
    return chain.with_retry(
        stop_after_attempt=100, wait_exponential_jitter=False
    ).batch(input_prompts, config={"max_concurrency": 4})


def evaluate_qa_chain(
        eval_df: pd.DataFrame,
        columns: List[str],
        chain_to_evaluate,
        run_name: str,
        llm_judge: str = "databricks-dbrx-instruct",
):
    """
    Runs the evaluation of the LangChain LCEL Chain using defined dataset with questions and answers and logs the results to MLflow
    :param eval_df: DataFrame which contains context, questions and answers
    :param columns: DataFrame columns which the chains require
    :param chain_to_evaluate: Chain to evaluate
    :param run_name: MLflow run name
    :return: result object provided by mlflow.evaluate
    """
    eval_df["prediction"] = run_chain_for_eval_data(
        chain_to_evaluate, eval_df[columns].to_dict(orient="records")
    )
    eval_df = eval_df.reset_index(drop=True).rename(columns={"question": "inputs"})

    with mlflow.start_run(run_name=run_name) as run:
        results = mlflow.evaluate(
            data=eval_df,
            targets="answer",
            predictions="prediction",
            extra_metrics=[
                mlflow.metrics.genai.answer_similarity(model=f"endpoints:/{llm_judge}")
            ],
            evaluators="default",
        )
        return results
