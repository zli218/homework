import json
import os

def save_to_jsonl(data_list: list, output_file: str):
    """将数据保存为 JSONL 格式，适用于 Qwen 微调"""
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for entry in data_list:
            if entry:
                # 转换为 Qwen 常见的 ChatML 格式或 Alpaca 格式
                # 这里我们使用通用的 Instruction 格式
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + "\n")
    print(f"已保存 {len(data_list)} 条数据到 {output_file}")
