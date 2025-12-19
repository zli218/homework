import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# === 配置区域 ===
# 使用 Qwen2.5-0.5B-Instruct，模型非常小，适合快速验证流程
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 
DATA_FILE = "qwen_finetune_data.jsonl"
OUTPUT_DIR = "./output_check"
MAX_STEPS = 30  # 仅跑 30 步用于验证，实际训练请调大

def main():
    # 0. 环境检查
    if not os.path.exists(DATA_FILE):
        print(f"错误：找不到数据文件 {DATA_FILE}")
        print("请先运行 main.py 生成数据。")
        return

    print(f">>> 正在加载 Tokenizer: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"模型下载或加载失败: {e}")
        print("提示: 国内用户请尝试设置环境变量 HF_ENDPOINT=https://hf-mirror.com")
        return

    # 1. 加载并处理数据
    print(f">>> 正在加载数据集...")
    try:
        dataset = load_dataset("json", data_files=DATA_FILE, split="train")
        print(f">>> 成功加载 {len(dataset)} 条样本。")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return
    
    def process_func(example):
        """将数据转换为模型输入格式"""
        MAX_LENGTH = 512 
        
        # 提取对话内容
        # 数据格式为 main.py 生成的 standard format
        msgs = example["conversation"]
        
        # 使用 chat template 格式化
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        
        # Tokenize
        model_inputs = tokenizer(
            text, 
            max_length=MAX_LENGTH, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = model_inputs.input_ids[0]
        attention_mask = model_inputs.attention_mask[0]
        
        # 简单构造 Labels: 让模型学习预测全文 (为了脚本简洁，不做复杂的 User Masking)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    print(">>> 正在预处理数据...")
    tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)

    # 2. 加载模型
    print(f">>> 正在加载模型 (这可能需要几分钟)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 配置 LoRA (轻量级微调)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=5,
        max_steps=MAX_STEPS, # 快速验证
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        save_strategy="no", 
        report_to="none"
    )

    # 5. 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    print(">>> 开始微调 (演示运行)...")
    trainer.train()
    
    print(f"\n>>> 验证完成！如果 Loss 在下降，说明数据格式被模型接受了。")

if __name__ == "__main__":
    main()