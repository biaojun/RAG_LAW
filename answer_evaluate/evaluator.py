from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    LLMContextRecall, 
    LLMContextPrecisionWithoutReference, 
    ContextEntityRecall, 
    NoiseSensitivity, 
    Faithfulness,
    ResponseRelevancy,
    MetricWithLLM,
    ContextRelevance,  # 新增
)
# 新增指标导入
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
import asyncio
import json
from enum import Enum
from typing import List, Union, Dict, Any
import sys
import numpy as np
import time  # 新增：用于计时
from tqdm.asyncio import tqdm_asyncio  # 新增：异步进度条
from tqdm import tqdm  # 新增：同步进度条

CONFIG_FILE = "config/config.json"

class LLMModel(Enum):
    ZHIPU = "ZHIPU"
    DEEPSEEK = "DEEPSEEK"

import copy
import typing as t
from ragas.prompt import PydanticPrompt
from typing import TypeVar

MetricType = TypeVar("MetricType", bound="MetricWithLLM")

def add_json_constraint_to_metric(metric: MetricType) -> MetricType:
    """
    Add JSON output constraints to ragas metric prompts, compatible with the set_prompts(** prompts) method.
    """
    def add_json_format_constraint(original_instruction: str) -> str:
        json_constraint = """

IMPORTANT OUTPUT REQUIREMENTS:
1. Return ONLY the valid JSON object — NO additional text, explanations, or comments.
2. Do NOT include markdown formatting (e.g., ```json, ```, code blocks).
3. Strict JSON syntax: double quotes, no trailing commas, properly closed brackets.
"""
        return original_instruction + json_constraint

    original_prompts: t.Dict[str, PydanticPrompt] = metric.get_prompts()
    updated_prompts = {}
    for prompt_name, original_prompt in original_prompts.items():
        updated_prompt = copy.deepcopy(original_prompt)
        updated_prompt.instruction = add_json_format_constraint(original_prompt.instruction)
        updated_prompts[prompt_name] = updated_prompt

    metric.set_prompts(**updated_prompts)
    return metric

class RAGEvaluator:
    def __init__(self, workdir = os.getcwd(), model = LLMModel.DEEPSEEK):        
        config_path = os.path.join(workdir, CONFIG_FILE)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_name = config[model.value]["model_name"]
        api_base = config[model.value]["api_base"]
        api_key = os.getenv(f"{model.value}_API_KEY")
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_base=api_base,
            openai_api_key=api_key,
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings(
            model="embedding-3",
            openai_api_base=config[LLMModel.ZHIPU.value]["api_base"],
            openai_api_key=os.getenv(f"{LLMModel.ZHIPU.value}_API_KEY")
        )

        # 初始化原有指标
        self.context_recall_metric = add_json_constraint_to_metric(LLMContextRecall(llm=self.llm))
        self.context_precision_metric = add_json_constraint_to_metric(LLMContextPrecisionWithoutReference(llm=self.llm))
        self.context_entity_recall_metric = add_json_constraint_to_metric(ContextEntityRecall(llm=self.llm))
        self.noise_sensitivity_metric = add_json_constraint_to_metric(NoiseSensitivity(llm=self.llm))
        self.faithfulness_metric = add_json_constraint_to_metric(Faithfulness(llm=self.llm))
        self.response_relevancy_metric = add_json_constraint_to_metric(ResponseRelevancy(
            llm=self.llm,
            embeddings=self.embeddings
        ))
        # self.context_relevance_metric = add_json_constraint_to_metric(ContextRelevance(llm=self.llm))
        

    # 原有指标计算方法添加进度条
    async def calculate_recall(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            # 为列表处理添加异步进度条
            return await tqdm_asyncio.gather(
                *[self.context_recall_metric.single_turn_ascore(s) for s in sample],
                desc="Calculating Context Recall"
            )
        else:
            return await self.context_recall_metric.single_turn_ascore(sample)

    async def calculate_precision(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            return await tqdm_asyncio.gather(
                *[self.context_precision_metric.single_turn_ascore(s) for s in sample],
                desc="Calculating Context Precision"
            )
        else:
            return await self.context_precision_metric.single_turn_ascore(sample)

    async def calculate_context_entity_recall(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            return await tqdm_asyncio.gather(
                *[self.context_entity_recall_metric.single_turn_ascore(s) for s in sample],
                desc="Calculating Entity Recall"
            )
        else:
            return await self.context_entity_recall_metric.single_turn_ascore(sample)

    async def calculate_noise_sensitivity(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            return await tqdm_asyncio.gather(
                *[self.noise_sensitivity_metric.single_turn_ascore(s) for s in sample],
                desc="Calculating Noise Sensitivity"
            )
        else:
            return await self.noise_sensitivity_metric.single_turn_ascore(sample)

    async def calculate_faithfulness(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            return await tqdm_asyncio.gather(
                *[self.faithfulness_metric.single_turn_ascore(s) for s in sample],
                desc="Calculating Faithfulness"
            )
        else:
            return await self.faithfulness_metric.single_turn_ascore(sample)

    async def calculate_response_relevancy(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            return await tqdm_asyncio.gather(
                *[self.response_relevancy_metric.single_turn_ascore(s) for s in sample],
                desc="Calculating Response Relevancy"
            )
        else:
            return await self.response_relevancy_metric.single_turn_ascore(sample)


    async def evaluate(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> dict:
        # 并行计算所有指标
        tasks = [
            self.calculate_recall(sample),
            self.calculate_precision(sample),
            self.calculate_context_entity_recall(sample),
            self.calculate_noise_sensitivity(sample),
            self.calculate_faithfulness(sample),
            self.calculate_response_relevancy(sample),
        ]
        # 同时执行所有任务
        recall_scores, precision_scores, entity_recall_scores, noise_sensitivity_scores, \
        faithfulness_scores, response_relevancy_scores = await asyncio.gather(*tasks)
    
        
        def ensure_list(value):
            return [value] if not isinstance(value, list) else value
            
        return {
            "context_recall": ensure_list(recall_scores),
            "context_precision": ensure_list(precision_scores),
            "entity_recall_scores": ensure_list(entity_recall_scores),
            "noise_sensitivity_scores": ensure_list(noise_sensitivity_scores),
            "faithfulness_scores": ensure_list(faithfulness_scores),
            "response_relevancy_scores": ensure_list(response_relevancy_scores),
        }
    
    async def evaluate_with_stats(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Dict[str, Any]:
        samples = sample if isinstance(sample, list) else [sample]
        total = len(samples)
        if total == 0:
            return {}

        # 记录评估开始时间
        start_time = time.time()
        
        eval_results = await self.evaluate(samples)
        
        # 计算评估耗时
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per sample: {elapsed_time/total:.4f} seconds" if total > 0 else "")
        
        metric_names = {
            "context_recall": "Context Recall",
            "context_precision": "Context Precision",
            "entity_recall_scores": "Entity Recall",
            "noise_sensitivity_scores": "Noise Sensitivity",
            "faithfulness_scores": "Faithfulness",
            "response_relevancy_scores": "Response Relevancy",
        }
        
        stats = {}
        
        # 为统计计算添加进度条
        for metric_key, scores in tqdm(eval_results.items(), desc="Calculating Statistics"):
            try:
                scores_array = np.array(scores, dtype=np.float64)
            except ValueError:
                scores_array = np.array([0.0])
            
            max_score = float(scores_array.max()) if len(scores_array) > 0 else 0.0
            min_score = float(scores_array.min()) if len(scores_array) > 0 else 0.0
            avg_score = float(scores_array.mean()) if len(scores_array) > 0 else 0.0
            
            max_indices = [i for i, s in enumerate(scores) if np.isclose(s, max_score)]
            min_indices = [i for i, s in enumerate(scores) if np.isclose(s, min_score)]
            
            def get_case_details(indices: List[int]) -> List[Dict[str, Any]]:
                return [
                    {
                        "index": i,
                        "user_input": samples[i].user_input,
                        "response": samples[i].response,
                        "reference": samples[i].reference,
                        "retrieved_contexts": samples[i].retrieved_contexts,
                        "score": round(float(scores[i]), 4)
                    } for i in indices
                ]
            
            stats[metric_key] = {
                "metric_name": metric_names.get(metric_key, metric_key),
                "max_score": round(max_score, 4),
                "min_score": round(min_score, 4),
                "average_score": round(avg_score, 4),
                "sample_count": total,
                "max_cases": get_case_details(max_indices),
                "min_cases": get_case_details(min_indices)
            }
        
        return stats

async def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluator.py xxx.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        # 加载数据时添加进度提示
        print(f"Loading data from {json_file}...")
        start_load = time.time()
        with open(json_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        load_time = time.time() - start_load
        print(f"Data loaded in {load_time:.2f} seconds")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file}' does not exist")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{json_file}' is not valid JSON format")
        sys.exit(1)
    
    # 转换样本时添加进度条
    samples = []
    for item in tqdm(test_data, desc="Preparing samples"):
        sample = SingleTurnSample(
            user_input=item.get("user_input", ""),
            response=item.get("response", ""),
            reference=item.get("reference", ""),
            retrieved_contexts=item.get("retrieved_contexts", [])
        )
        samples.append(sample)
    
    # 总耗时统计
    total_start_time = time.time()
    
    evaluator = RAGEvaluator()
    stats_result = await evaluator.evaluate_with_stats(samples)
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal processing time: {total_elapsed:.2f} seconds")
    
    print("Evaluation Statistics:")
    print(json.dumps(stats_result, ensure_ascii=False, indent=2))
    
    output_file = "evaluation_stats.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats_result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())