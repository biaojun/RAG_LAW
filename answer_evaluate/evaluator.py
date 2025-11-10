from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    LLMContextRecall, 
    LLMContextPrecisionWithoutReference, 
    ContextEntityRecall, 
    NoiseSensitivity, 
    Faithfulness,
    ResponseRelevancy,
    MetricWithLLM,
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
import asyncio
import json
from enum import Enum
from typing import List, Union, Dict, Any
import sys
import numpy as np  # New addition: for more stable numerical calculations

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
    # Define JSON format constraints
    def add_json_format_constraint(original_instruction: str) -> str:
        json_constraint = """

IMPORTANT OUTPUT REQUIREMENTS:
1. Return ONLY the valid JSON object — NO additional text, explanations, or comments.
2. Do NOT include markdown formatting (e.g., ```json, ```, code blocks).
3. Strict JSON syntax: double quotes, no trailing commas, properly closed brackets.
"""
        return original_instruction + json_constraint

    # 1. Get original prompts (dictionary format: {prompt_name: PydanticPrompt, ...})
    original_prompts: t.Dict[str, PydanticPrompt] = metric.get_prompts()

    # 2. Process each prompt and update the instruction
    updated_prompts = {}
    for prompt_name, original_prompt in original_prompts.items():
        # Deep copy to avoid modifying the original instance
        updated_prompt = copy.deepcopy(original_prompt)
        # Append format constraints
        updated_prompt.instruction = add_json_format_constraint(original_prompt.instruction)
        # Store in the updated dictionary
        updated_prompts[prompt_name] = updated_prompt

    # 3. Key correction: convert the dictionary to keyword arguments and pass to set_prompts
    metric.set_prompts(**updated_prompts)  # Use** to unpack the dictionary into keyword arguments

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

        self.context_recall_metric = add_json_constraint_to_metric(LLMContextRecall(llm=self.llm))
        self.context_precision_metric = add_json_constraint_to_metric(LLMContextPrecisionWithoutReference(llm=self.llm))
        self.context_entity_recall_metric = add_json_constraint_to_metric(ContextEntityRecall(llm=self.llm))
        self.noise_sensitivity_metric = add_json_constraint_to_metric(NoiseSensitivity(llm=self.llm))
        self.faithfulness_metric = add_json_constraint_to_metric(Faithfulness(llm=self.llm))
        self.response_relevancy_metric = add_json_constraint_to_metric(ResponseRelevancy(
            llm=self.llm,
            embeddings=self.embeddings
        ))

    async def calculate_recall(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            results = []
            for s in sample:
                score = await self.context_recall_metric.single_turn_ascore(s)
                results.append(score)
            return results
        else:
            return await self.context_recall_metric.single_turn_ascore(sample)

    async def calculate_precision(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            results = []
            for s in sample:
                score = await self.context_precision_metric.single_turn_ascore(s)
                results.append(score)
            return results
        else:
            return await self.context_precision_metric.single_turn_ascore(sample)

    async def calculate_context_entity_recall(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            results = []
            for s in sample:
                score = await self.context_entity_recall_metric.single_turn_ascore(s)
                results.append(score)
            return results
        else:
            return await self.context_entity_recall_metric.single_turn_ascore(sample)

    async def calculate_noise_sensitivity(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            results = []
            for s in sample:
                score = await self.noise_sensitivity_metric.single_turn_ascore(s)
                results.append(score)
            return results
        else:
            return await self.noise_sensitivity_metric.single_turn_ascore(sample)

    async def calculate_faithfulness(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            results = []
            for s in sample:
                score = await self.faithfulness_metric.single_turn_ascore(s)
                results.append(score)
            return results
        else:
            return await self.faithfulness_metric.single_turn_ascore(sample)

    async def calculate_response_relevancy(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> Union[float, List[float]]:
        if isinstance(sample, list):
            results = []
            for s in sample:
                score = await self.response_relevancy_metric.single_turn_ascore(s)
                results.append(score)
            return results
        else:
            return await self.response_relevancy_metric.single_turn_ascore(sample)

    async def evaluate(self, sample: Union[SingleTurnSample, List[SingleTurnSample]]) -> dict:
        # Fix: Remove extra commas and ensure all results are list types
        recall_scores = await self.calculate_recall(sample)
        precision_scores = await self.calculate_precision(sample)
        entity_recall_scores = await self.calculate_context_entity_recall(sample)
        noise_sensitivity_scores = await self.calculate_noise_sensitivity(sample)
        faithfulness_scores = await self.calculate_faithfulness(sample)
        response_relevancy_scores = await self.calculate_response_relevancy(sample)
        
        # Unify format: Ensure lists are returned even for single samples
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
        """
        Extended evaluation function that calculates max, min, average of each score, and outputs corresponding cases
        """
        # Ensure input is in list form
        samples = sample if isinstance(sample, list) else [sample]
        total = len(samples)
        if total == 0:
            return {}

        # Get original evaluation results
        eval_results = await self.evaluate(samples)
        
        # Define mapping between metric names and display names for better readability
        metric_names = {
            "context_recall": "Context Recall",
            "context_precision": "Context Precision",
            "entity_recall_scores": "Entity Recall",
            "noise_sensitivity_scores": "Noise Sensitivity",
            "faithfulness_scores": "Faithfulness",
            "response_relevancy_scores": "Response Relevancy"
        }
        
        # Initialize statistics result dictionary
        stats = {}
        
        # Process each metric
        for metric_key, scores in eval_results.items():
            # Convert to numpy array for stable calculations (handle potential numeric type inconsistencies)
            try:
                scores_array = np.array(scores, dtype=np.float64)
            except ValueError:
                # Handle conversion failure
                scores_array = np.array([0.0])
            
            # Calculate statistics (using numpy for stability)
            max_score = float(scores_array.max()) if len(scores_array) > 0 else 0.0
            min_score = float(scores_array.min()) if len(scores_array) > 0 else 0.0
            avg_score = float(scores_array.mean()) if len(scores_array) > 0 else 0.0
            
            # Find indices of cases with max/min scores
            max_indices = [i for i, s in enumerate(scores) if np.isclose(s, max_score)]
            min_indices = [i for i, s in enumerate(scores) if np.isclose(s, min_score)]
            
            # Helper function to collect case details
            def get_case_details(indices: List[int]) -> List[Dict[str, Any]]:
                return [
                    {
                        "index": i,  # Index of the case in the input list
                        "user_input": samples[i].user_input,
                        "response": samples[i].response,
                        "reference": samples[i].reference,
                        "retrieved_contexts": samples[i].retrieved_contexts,
                        "score": round(float(scores[i]), 4)  # Ensure uniform score format
                    } for i in indices
                ]
            
            # Build statistics result for this metric
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
    """
    Main function to execute the RAG evaluation process.
    
    Usage:
        python evaluator.py <input_json_file>
        
    Arguments:
        <input_json_file> - Path to the JSON file containing test data. 
                            The JSON file should be a list of objects, each containing:
                            - user_input (str): The user's query
                            - response (str): The generated response to evaluate
                            - reference (str): The reference answer (ground truth)
                            - retrieved_contexts (list): List of retrieved context strings used for generating the response
                            
    Output:
        - Prints evaluation statistics to console
        - Saves evaluation results to "evaluation_stats.json"
    """
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python evaluator.py xxx.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # Read JSON file content
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' does not exist")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{json_file}' is not valid JSON format")
        sys.exit(1)
    
    # Convert to list of SingleTurnSample
    samples = []
    for item in test_data:
        # Create sample
        sample = SingleTurnSample(
            user_input=item.get("user_input", ""),
            response=item.get("response", ""),
            reference=item.get("reference", ""),
            retrieved_contexts=item.get("retrieved_contexts", [])
        )
        samples.append(sample)
    
    # Execute evaluation and calculate statistics
    evaluator = RAGEvaluator()
    stats_result = await evaluator.evaluate_with_stats(samples)
    
    # Output formatted results
    print("Evaluation Statistics:")
    print(json.dumps(stats_result, ensure_ascii=False, indent=2))
    
    # Save results to file
    output_file = "evaluation_stats.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats_result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())