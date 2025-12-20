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


# 使用 Qwen2.5-0.5B-Instruct，模型非常小，适合快速验证流程
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 
DATA_FILE = "qwen_finetune_data.jsonl"
OUTPUT_DIR = "./output_check"
MAX_STEPS = 30  

def main():
    #  环境检查
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

    #  加载并处理数据
    print(f">>> 正在加载数据集...")
    try:
        full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
        print(f">>> 成功加载 {len(full_dataset)} 条样本。")
        
        # 切分训练集和验证集 (9:1)
        if len(full_dataset) >= 10:
            split_ds = full_dataset.train_test_split(test_size=0.1, seed=42)
            train_ds = split_ds["train"]
            eval_ds = split_ds["test"]
            print(f">>> 训练集: {len(train_ds)} 条, 验证集: {len(eval_ds)} 条")
        else:
            train_ds = full_dataset
            eval_ds = None
            print(">>> 样本过少，跳过验证集切分。")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return
    
    def process_func(example):
        """将数据转换为模型输入格式"""
        MAX_LENGTH = 512 
        
        # 提取对话内容
        # 数据格式为 main.py 生成的 standard format
        msgs = example["conversation"]
        
       
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
        
        # 简单构造 Labels
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    print(">>> 正在预处理数据...")
    tokenized_train = train_ds.map(process_func, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(process_func, remove_columns=eval_ds.column_names) if eval_ds else None

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
        logging_steps=1,
        max_steps=MAX_STEPS, # 快速验证
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        # 添加验证配置
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        load_best_model_at_end=True if eval_ds else False,
        report_to="none"
    )

    # 5. 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    print("\n>>> 开始微调 (演示运行)...")
    trainer.train()
    
    print(f"\n>>> 训练完成！")

if __name__ == "__main__":
    main()