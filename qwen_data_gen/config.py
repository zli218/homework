import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    # API 配置
    API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini")  # Teacher Model

    # 目标代码仓配置
    REPO_URL = "https://github.com/cosmicpython/code.git"
    LOCAL_REPO_PATH = "./temp_repo"
    
    # 数据生成配置
    OUTPUT_FILE = "qwen_finetune_data.jsonl"
    MAX_SAMPLES = 100 # 样本数量
    
    # 多样性配置 (Temperature Scaling)
    MIN_TEMP = 0.7
    MAX_TEMP = 0.9

    # RAG 配置
    RAG_ENABLED = False
    RAG_TOP_K = 2

    # 自我修正配置
    CRITIC_ENABLED = True
    CRITIC_THRESHOLD = 7  # 低于此分数的样本将被丢弃

    # 中英文生成比例配置 (0.0 to 1.0)
    ENGLISH_RATIO = 0.3

    if not API_KEY:
        raise ValueError("请在环境变量或 .env 文件中设置 GOOGLE_API_KEY")
