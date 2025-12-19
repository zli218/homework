import json
import random
import sys
import uuid
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config

class DataGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=Config.API_KEY)
        
        # Persona Rotation: 多样化人设
        self.personas = [
            "资深 DDD 架构师，擅长领域驱动设计和解耦。",
            "Python 性能优化专家，关注代码效率和算法复杂度。",
            "高级后端工程师，注重代码的可读性、健壮性和异常处理。",
            "安全审计专家，关注潜在的代码漏洞和数据安全。"
        ]

    def _get_random_persona(self):
        return random.choice(self.personas)

    def _get_random_temp(self):
        return random.uniform(Config.MIN_TEMP, Config.MAX_TEMP)

    def _format_assistant_content(self, output_content):
        """将助手的输出（可能是字符串或字典）格式化为统一的字符串"""
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

    def _extract_clean_instruction(self, raw_data):
        """清洗 LLM 返回的 instruction，防止包含 Prompt 模板"""
        instruction = raw_data.get("instruction", "")
        output = raw_data.get("output")

        # 如果 instruction 看起来像 Prompt 模板，尝试从 output 恢复
        if "你现在扮演" in instruction or "任务：" in instruction:
            if isinstance(output, dict):
                for key in ["requirement", "new_requirement", "question"]:
                    if key in output and output[key]:
                        return output[key]
        
        return instruction

    def _build_qa_prompt(self, chunk: dict, persona: str) -> str:
        return f"""
        你现在扮演一名：{persona}
        
        请分析以下 Python 代码片段（来自文件: {chunk['file_name']}）：
        
        ```python
        {chunk['code']}
        ```
        
        任务：
        1. 针对这段代码的业务逻辑或技术实现，提出了一个有深度的技术问题。
        2. 提供详细的推理过程 (Reasoning Trace)，解释你是如何分析代码并得出答案的。
        3. 给出最终答案。
        
        请严格以 JSON 格式输出，不要包含 Markdown 标记，格式如下：
        {{
            "instruction": "你的问题...",
            "input": "代码片段...",
            "output": "你的回答（包含推理过程）..."
        }}
        
        注意：Output 字段中必须显式包含 '【推理过程】' 和 '【最终答案】' 两个部分。
        """

    def _build_arch_prompt(self, chunk: dict, persona: str) -> str:
        return f"""
        你现在扮演一名：{persona}
        
        基于以下现有的代码逻辑（来自文件: {chunk['file_name']}）：
        
        ```python
        {chunk['code']}
        ```
        
        任务：
        1. 假设业务方提出了一个新的需求（例如：增加缓存、改为异步处理、增加审计日志等，请自行构思一个合理需求）。
        2. 设计一个基于当前代码的架构修改方案。
        3. 提供详细的推理过程 (Reasoning Trace)，解释为什么选择这个方案，以及它如何符合 DDD 或高质量代码原则。
        
        请严格以 JSON 格式输出，不要包含 Markdown 标记，格式如下：
        {{
            "instruction": "新需求描述...",
            "input": "当前代码片段...",
            "output": "架构设计方案（包含推理过程）..."
        }}
        """

    # 优化指数退避策略：初始等待至少 10秒，每次失败等待时间翻倍 (multiplier=2)，最大等待 120秒
    @retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, min=10, max=120))
    def generate_sample(self, chunk: dict, scenario: str = "qa") -> dict:
        persona = self._get_random_persona()
        
        if scenario == "qa":
            prompt = self._build_qa_prompt(chunk, persona)
        else:
            prompt = self._build_arch_prompt(chunk, persona)

        try:
            # 使用 Google 原生 SDK 生成内容
            # 我们将 System Instruction 拼接到 Prompt 前面，并启用 JSON 模式
            full_prompt = f"你是一个负责生成高质量代码微调数据的助手。请只输出 JSON。\n\n{prompt}"
            
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=self._get_random_temp(),
                    response_mime_type="application/json"
                )
            )
        except Exception as e:
            if "404" not in str(e) and "Not Found" not in str(e):
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"\n[Warning] 触发 Google 限流 (429)，正在等待重试... (详情: {str(e)[:50]}...)")
                raise e
            print(f"\n[Error] API 调用失败 (404): 找不到模型或 API Key 无效。")
            print(f"详细错误信息: {e}")
            print(f"请检查 .env 文件：\n1. GOOGLE_API_KEY 是否是真实的？(不要使用示例 Key)\n2. MODEL_NAME 是否支持？(当前: {Config.MODEL_NAME})")
            sys.exit(1) # 配置错误直接退出，避免空跑

        content = response.text
        try:
            raw_data = json.loads(content)
            
            # 转换数据格式为 Qwen 微调标准格式
            scenario_mapped = "business_logic" if scenario == "qa" else "architecture_design"
            user_content = self._extract_clean_instruction(raw_data)
            if raw_data.get("input"):
                user_content += f"\n\n相关代码:\n```python\n{raw_data.get('input')}\n```"

            new_data = {
                "id": str(uuid.uuid4()),
                "scenario": scenario_mapped,
                "code_ref": chunk["file_name"],
                "conversation": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": self._format_assistant_content(raw_data.get("output"))}
                ]
            }
            return new_data
        except json.JSONDecodeError:
            print(f"JSON 解析失败，跳过该条目。原始内容: {content[:100]}...")
            return None
