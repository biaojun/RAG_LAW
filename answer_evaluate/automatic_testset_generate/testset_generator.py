import json
import os
from typing import List, Dict, Any
from datasets import Dataset
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from zai import ZhipuAiClient

class LegalTestsetGenerator:
    def __init__(self):
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_base="https://api.deepseek.com/v1", 
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0
        )
        embedding = OpenAIEmbeddings(
                model="embedding-3",
                client=ZhipuAiClient(api_key= self.zhipu_api_key)
            )
        self.generator = TestsetGenerator(
            llm=llm,
            embedding_model=embedding,
        )

    
    from langchain_core.documents import Document  # 导入Document类

# 在load_legal_documents方法中修改文档构建部分
    def load_legal_documents(self, json_file_path: str) -> List[Document]:
        """
        加载法律案例文档，转换为LangChain Document格式
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        documents = []
        for case in cases:
            doc_content = self._build_document_content(case)
            # 直接创建LangChain Document对象
            document = Document(
                page_content=doc_content,
                metadata={
                    "id": case.get("id", ""),
                    "category": case.get("category", ""),
                    "crime_type": case.get("criminal_charge", ""),
                    "case_name": case.get("case_name", ""),
                    "keywords": case.get("keywords", []),
                    "judgment_date": case.get("judgment_date", "")
                }
            )
            documents.append(document)
        
        print(f"成功加载 {len(documents)} 个法律案例文档")
        return documents

    def _build_document_content(self, case: Dict) -> str:
        """构建文档内容"""
        content_parts = [
            f"案件名称: {case.get('case_name', '无名案件')}",
            f"案件类别: {case.get('category', '')}",
            f"涉嫌罪名: {case.get('criminal_charge', '')}",
            f"判决日期: {case.get('judgment_date', '')}",
            "",
            "【案件事实】",
            case.get('case_facts', '无'),
            "",
            "【判决理由】", 
            case.get('judgment_reason', '无'),
            "",
            "【裁判要旨】",
            case.get('judgement_holding', '无'),
            "",
            "【关键词】",
            ", ".join(case.get('keywords', [])),
            "",
            "【相关法律】",
            self._format_related_laws(case.get('related_laws', []))
        ]
        
        return "\n".join(content_parts)
    
    def _format_related_laws(self, laws: List[Dict]) -> str:
        """格式化相关法律信息"""
        if not laws:
            return "无"
        
        law_texts = []
        for law in laws:
            law_name = law.get('law_name', '')
            articles = law.get('article_numbers', [])
            articles_text = "、".join(articles)
            law_texts.append(f"{law_name} 第{articles_text}条")
        
        return "; ".join(law_texts)
    
    def generate_testset(self, documents: List[Document], test_size: int = 25) -> Dataset:
        """
        生成测试集
        
        Args:
            documents: 文档列表
            test_size: 测试集大小
            
        Returns:
            Dataset: 生成的测试集
        """
        print("开始生成测试集...")
        
        # 生成测试集
        testset = self.generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=test_size,
        )
        
        print(f"成功生成 {len(testset)} 个测试样本")
        return testset
    
    def save_testset(self, testset: List[Dict], output_path: str):
        """
        保存测试集到文件
        
        Args:
            testset: 测试集
            output_path: 输出文件路径
        """
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(testset, f, ensure_ascii=False, indent=2)
        
        print(f"测试集已保存到: {output_path}")
    
    def _classify_difficulty(self, question: str) -> str:
        """根据问题内容分类难度"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['是否违法', '会不会犯罪', '违法吗', '犯罪吗']):
            return "easy"
        elif any(word in question_lower for word in ['构成什么罪', '量刑', '处罚', '刑事责任']):
            return "medium"
        elif any(word in question_lower for word in ['对比', '区别', '如何认定', '法律适用']):
            return "hard"
        else:
            return "medium"
def main():
    """主函数"""
    
    # 配置参数
    INPUT_JSON = "testdataset.json"
    OUTPUT_FILE = "legal_rag_testset.json"
    
    try:
        # 1. 初始化生成器
        print("初始化法律测试集生成器...")
        legal_generator = LegalTestsetGenerator()
        
        # 2. 加载法律文档
        print("加载法律案例文档...")
        documents = legal_generator.load_legal_documents(INPUT_JSON)
        
        # 3. 生成测试集
        print("使用Ragas生成测试集...")
        testset = legal_generator.generate_testset(documents, test_size=20)
        
        # 4. 保存测试集
        print("保存测试集...")
        legal_generator.save_testset(testset.to_list(), OUTPUT_FILE)
        
        print("✅ 法律RAG测试集生成完成！")
        
    except Exception as e:
        print(f"❌ 使用Ragas生成测试集失败: {e}")

if __name__ == "__main__":
    main()