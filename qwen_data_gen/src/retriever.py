import numpy as np
from typing import List, Dict
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None

class CodeRetriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.model = None
        self.embeddings = None
        
        if SentenceTransformer:
            print(">>> 正在初始化 RAG 检索模型 (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.build_index()
        else:
            print("[Warning] 未安装 sentence-transformers，RAG 功能将不可用。")

    def build_index(self):
        if not self.model or not self.chunks:
            return
        
        print(f">>> 正在构建代码索引 (共 {len(self.chunks)} 个片段)...")
        # 简单的将文件名和代码拼接作为索引文本
        texts = [f"File: {c['file_name']}\n{c['code']}" for c in self.chunks]
        self.embeddings = self.model.encode(texts)

    def retrieve(self, query_chunk: Dict, k: int = 2) -> List[Dict]:
        if self.model is None or self.embeddings is None:
            return []

        query_text = f"File: {query_chunk['file_name']}\n{query_chunk['code']}"
        query_vec = self.model.encode([query_text])
        
        # 计算余弦相似度
        sim_matrix = cosine_similarity(query_vec, self.embeddings)[0]
        
        # 获取 Top K 索引 (排除自身)
        # argsort 返回的是从小到大的索引，所以取反切片
        sorted_indices = np.argsort(sim_matrix)[::-1]
        
        results = []
        for idx in sorted_indices:
            if len(results) >= k:
                break
            
            candidate = self.chunks[idx]
            # 简单的去重逻辑：排除完全相同的代码片段
            if candidate['file_name'] == query_chunk['file_name'] and candidate['code'] == query_chunk['code']:
                continue
                
            results.append(candidate)
            
        return results