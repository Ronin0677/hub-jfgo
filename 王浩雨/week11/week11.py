import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig

# 模型路径
MODEL_DIR = "./models/Qwen2.5-0.5B-Instruct"
# 输出目录
OUTPUT_DIR = "./output"
# 最大序列长度
MAX_SEQ_LEN = 512

def create_instruction_dataset():
    """
    创建包含中文指令的训练数据集。
    """
    data = [
        {
            "instruction": "你们的客服工作时间是？",
            "input": "",
            "output": "我们的在线客服团队工作时间为周一至周日，每日上午9点至晚上21点（夏令时为上午8点至晚上20点）。您可以随时联系我们获取帮助。"
        },
        {
            "instruction": "配送时间是多久？",
            "input": "",
            "output": "亲爱的顾客，我们的标准配送时间为3-5个工作日内送达，但具体的时间会根据您所在地区的物流情况和包裹重量有所变动。详情请参考您订单页面的预计到达日期。"
        },
        {
            "instruction": "退货的时间限制是多久？",
            "input": "",
            "output": "我们接受的无理由退货时间为购买后7天内。如果您需要因商品质量问题进行退货，此时间从收到并确认问题后的次日开始计算。超过时间范围将无法受理。"
        },
        {
            "instruction": "是否有货?",
            "input": "",
            "output": "能下单就是在售有货的哦亲亲"
        },
        {
            "instruction": "能否开具发票？",
            "input": "",
            "output": "是的，我们可以为所有购买的商品提供正规发票服务。如果您需要发票，请在订单提交时留下相关信息或联系我们的客服进行后续操作。"
        },
    ]

    # 格式化数据为 SFTTrainer 所需的文本格式
    formatted_data = []
    for item in data:
        text_format = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted_data.append({"text": text_format})

    return formatted_data

def load_model_and_tokenizer():
    """
    加载预训练模型和分词器。
    """
    print(f"加载模型: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # 设置pad_token，如果分词器没有定义的话
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main():
    """
    主函数，执行模型微调。
    """
    print("=" * 50)
    print("开始SFT微调训练")
    print("=" * 50)

    model, tokenizer = load_model_and_tokenizer()

    print("\n构造训练数据...")
    train_data_list = create_instruction_dataset()
    train_dataset = Dataset.from_list(train_data_list)

    print(f"训练样本数量: {len(train_dataset)}")

    # 配置训练参数
    trainer_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        report_to=None,
        dataloader_pin_memory=False,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
    )

    # 初始化SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=trainer_config,
        train_dataset=train_dataset,
        processing_class=tokenizer, # 这里将tokenizer作为processing_class传递，虽然通常是Tokenizer类
    )

    print("\n开始训练...")
    trainer.train()

    print(f"\n训练完成，正在保存模型到 {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 50)
    print("训练完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
