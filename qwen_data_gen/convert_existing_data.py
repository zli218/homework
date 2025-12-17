import json
import uuid
import os

# 配置输入输出文件名
INPUT_FILE = "qwen_finetune_data.jsonl"
OUTPUT_FILE = "qwen_finetune_data_converted.jsonl"

def format_assistant_content(output_content):
    """
    将助手的输出（可能是字符串或字典）格式化为统一的字符串。
    逻辑复用自 generator.py
    """
    if isinstance(output_content, str):
        return output_content
    elif isinstance(output_content, dict):
        trace = output_content.get("reasoning_trace", "")
        if isinstance(trace, (dict, list)):
            trace = json.dumps(trace, ensure_ascii=False, indent=2)
        
        plan = output_content.get("architectural_modification_plan") or \
               output_content.get("output") or \
               {k: v for k, v in output_content.items() if k != "reasoning_trace"}
        
        if isinstance(plan, (dict, list)):
            plan = json.dumps(plan, ensure_ascii=False, indent=2)
            
        return f"推理过程 (Reasoning Trace):\n{trace}\n\n最终答案:\n{plan}"
    return str(output_content)

def extract_clean_instruction(data):
    """
    从数据中提取真正的用户指令。
    如果 'instruction' 字段包含了 Prompt 模板（如 '你现在扮演...'），则尝试从 'output' 中找回真正的需求。
    """
    instruction = data.get("instruction", "")
    output = data.get("output")

    # 启发式检查：如果指令包含 Prompt 的特征词，说明是脏数据
    if "你现在扮演" in instruction or "任务：" in instruction or "Task:" in instruction:
        if isinstance(output, dict):
            # 尝试从 output 中提取真正的需求描述 (常见于架构设计场景)
            for key in ["requirement", "new_requirement", "instruction", "question", "problem_statement"]:
                if key in output and output[key]:
                    return output[key]
    return instruction

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    
    converted_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告：第 {line_num} 行 JSON 解析失败，跳过。")
                continue

            # 检查是否已经是新格式 (包含 conversation 字段)
            if "conversation" in data:
                # 直接写入新文件，保持原样
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # 开始转换旧格式
            meta = data.get("meta", {})
            
            # 1. 映射场景
            old_scenario = meta.get("scenario", "qa")
            new_scenario = "business_logic" if old_scenario == "qa" else "architecture_design"
            
            # 2. 构建 User Content (将代码拼接到 Instruction 后)
            user_content = extract_clean_instruction(data)
            code_input = data.get("input", "")
            if code_input:
                user_content += f"\n\n相关代码:\n```python\n{code_input}\n```"
            
            # 3. 组装新对象
            new_entry = {
                "id": str(uuid.uuid4()),
                "scenario": new_scenario,
                "code_ref": meta.get("source_file", "unknown_file.py"),
                "conversation": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": format_assistant_content(data.get("output"))}
                ]
            }
            
            f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            converted_count += 1

    print(f"转换完成！")
    print(f"- 成功转换旧数据: {converted_count} 条")
    print(f"- 结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()