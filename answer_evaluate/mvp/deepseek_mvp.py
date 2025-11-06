from ragas import evaluate
from ragas.metrics import faithfulness
from ragas import EvaluationDataset
from langchain_openai import ChatOpenAI
import os

def minimal_test_fixed():
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1", 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0
    )
    
    dataset = EvaluationDataset.from_dict([{
        'user_input': '什么是AI？',                    
        'response': 'AI是人工智能的缩写。',           
        'retrieved_contexts': ['人工智能是计算机科学的一个分支。']  
    }])
    
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=llm
        )
        print(f"极简测试成功: {result}")
    except Exception as e:
        print(f"极简测试失败: {e}")

minimal_test_fixed()