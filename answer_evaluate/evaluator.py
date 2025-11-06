from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    LLMContextRecall, 
    LLMContextPrecisionWithoutReference, 
    ContextEntityRecall, 
    NoiseSensitivity, 
    Faithfulness,
    ResponseRelevancy  # 新增导入
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings  # 新增嵌入模型导入
import os
import asyncio
import json
from enum import Enum
from typing import List, Union

CONFIG_FILE = "config/config.json"

class LLMModel(Enum):
    ZHIPU = "ZHIPU"
    DEEPSEEK = "DEEPSEEK"
    


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

        self.context_recall_metric = LLMContextRecall(llm=self.llm)
        self.context_precision_metric = LLMContextPrecisionWithoutReference(llm=self.llm)
        self.context_entity_recall_metric = ContextEntityRecall(llm=self.llm)
        self.noise_sensitivity_metric = NoiseSensitivity(llm=self.llm)
        self.faithfulness_metric = Faithfulness(llm=self.llm)
        self.response_relevancy_metric = ResponseRelevancy(
            llm=self.llm,
            embeddings=self.embeddings
        )

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
        recall_scores = await self.calculate_recall(sample)
        precision_scores = await self.calculate_precision(sample)
        entity_recall_scores = await self.calculate_context_entity_recall(sample)
        noise_sensitivity_scores = await self.calculate_noise_sensitivity(sample)
        faithfulness_scores = await self.calculate_faithfulness(sample)
        response_relevancy_scores = await self.calculate_response_relevancy(sample)
        return {
            "context_recall": recall_scores,
            "context_precision": precision_scores,
            "context_entity_recall": entity_recall_scores,
            "noise_sensitivity": noise_sensitivity_scores,
            "faithfulness": faithfulness_scores,
            "response_relevancy": response_relevancy_scores 
        }

async def main():
    evaluator = RAGEvaluator()
    
    single_sample = SingleTurnSample(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["The Eiffel Tower is located in Paris."],
    )
    
    sample2 = SingleTurnSample(
        user_input="What is the height of Eiffel Tower?",
        response="It is about 324 meters tall.",
        reference="The Eiffel Tower stands at 324 meters in height.",
        retrieved_contexts=[
            "The Eiffel Tower is a famous landmark in Paris.", 
            "It is 324 meters high." 
        ],
    )
    
    evaluation_result = await evaluator.evaluate([single_sample, sample2])
    print(f"综合评估结果: {evaluation_result}")


if __name__ == "__main__":
    asyncio.run(main())