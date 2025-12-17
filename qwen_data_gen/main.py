import random
from config import Config
from src.repo_loader import RepoLoader
from src.code_parser import CodeParser
from src.generator import DataGenerator
from src.utils import save_to_jsonl

try:
    from tqdm import tqdm
except ImportError:
    # 如果未安装 tqdm，使用一个简单的透传函数作为回退，避免报错
    def tqdm(iterable, *args, **kwargs):
        return iterable

def main():
    print(">>> 启动 Qwen 数据生成流水线...")
    
    # 1. Clone 代码仓
    loader = RepoLoader(Config.REPO_URL, Config.LOCAL_REPO_PATH)
    loader.clone_repo()
    files = loader.get_python_files()
    print(f">>> 发现 {len(files)} 个 Python 文件。")

    # 2. 解析代码
    parser = CodeParser()
    all_chunks = []
    for file_path in files:
        content = loader.read_file(file_path)
        file_name = file_path.split("/")[-1]
        chunks = parser.parse_code(content, file_name)
        all_chunks.extend(chunks)
    
    print(f">>> 提取了 {len(all_chunks)} 个代码片段 (Class/Function)。")
    
    # 3. 生成数据
    generator = DataGenerator()
    generated_data = []
    
    # 随机采样以避免 Token 消耗过多，实际生产可全量运行
    target_chunks = random.sample(all_chunks, min(Config.MAX_SAMPLES, len(all_chunks)))
    
    print(">>> 开始调用 Qwen-Plus 生成微调数据...")
    for chunk in tqdm(target_chunks):
        # 随机选择场景：QA 或 架构设计
        scenario = random.choice(["qa", "architecture"])
        
        result = generator.generate_sample(chunk, scenario=scenario)
        if result:
            generated_data.append(result)
            
            # 实时保存，防止中断丢失
            if len(generated_data) % 5 == 0:
                save_to_jsonl(generated_data[-5:], Config.OUTPUT_FILE)

    # 保存剩余数据
    remaining = len(generated_data) % 5
    if remaining > 0:
        save_to_jsonl(generated_data[-remaining:], Config.OUTPUT_FILE)

    print(f">>> 任务完成！共生成 {len(generated_data)} 条高质量数据。")
    print(f">>> 数据已保存至: {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()
