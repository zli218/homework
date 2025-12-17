import os
from dotenv import load_dotenv
from google import genai

# 加载环境变量
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")

print(f"--- 连接测试 ---")
print(f"API Key (前10位): {api_key[:10] if api_key else '未找到'}")
print(f"目标模型: {model_name}")

if not api_key or "AIzaSy" in api_key and len(api_key) < 40: 
    # 简单的启发式检查，防止用户使用占位符
    print("\n[警告] 你似乎正在使用示例/占位符 API Key。")
    print("请务必去 https://aistudio.google.com/ 申请一个真实的 Key 并填入 .env 文件。")

try:
    client = genai.Client(api_key=api_key)
    print("\n正在尝试列出可用模型 (List Models)...")
    # 尝试列出模型，这能验证 Key 是否有效
    pager = client.models.list()
    print("验证成功！你的账号支持以下模型：")
    for model in pager:
        if "gemini" in model.name:
            print(f" - {model.name}")
except Exception as e:
    print(f"\n[连接失败] 详细错误:\n{e}")
    print("\n如果错误包含 '400' 或 '404'，通常意味着 API Key 无效。")