#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_complete_llama.py

• 语法/路径/设备检测完善
• 明确设置 pad_token、防止 attention_mask 警告
• 支持批量多轮对话测试
• 保留模型生成的所有行控制指令
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts_data.prompts_data import prompt

# ─────────────────── 基本配置 ───────────────────
MODEL_DIR = "models/VCU-test"   # ← 如有需要请自行修改
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HF_TOKEN  = os.environ.get("HF_TOKEN")                              # 在环境变量里放入 Hugging Face 令牌

# ─────────────────── 载入模型和分词器 ───────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_TOKEN)

# LLAMA 默认没有 <pad>，我们补一个，避免 “pad 与 eos 相同” 警告
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, token=HF_TOKEN)

# 这里新增 mean_resizing=False ↓
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

model.to(DEVICE)

# 如需减少显存占用，可开启动态量化（速度可能稍升，精度需自测）
# model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# ).to(DEVICE)

# ─────────────────── 测试对话 ───────────────────
prompts=prompt
# ─────────────────── 推理函数 ───────────────────
def generate_control_plan(dialog: dict) -> str:
    """
    把形如 {'conversations':[...]} 的对话交给模型，
    返回模型生成的完整控制指令（可能包含多行）。
    """
    # 1. 转换为 OpenAI 风格 role 格式
    messages = [
        {"role": item["from"], "content": item["value"]}
        for item in dialog["conversations"]
    ]

    # 2. 用 tokenizer 模板拼接＋截断
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        add_generation_prompt=True   # 在末尾自动补一个 <assistant>
    ).to(DEVICE)

    # 3. attention_mask 显式设置，避免警告
    attention_mask = (input_ids != tokenizer.pad_token_id).to(DEVICE)

    # 4. 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 5. 取出新生成部分 → 解码
    new_tokens = output_ids[0, input_ids.shape[-1]:]
    raw_resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 6. 清理：去空行；去掉可能的 "assistant:" 前缀；保留所有行
    lines = [ln.strip() for ln in raw_resp.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("assistant"):
        lines = lines[1:]

    return "\n".join(lines)

# ─────────────────── 主程序 ───────────────────
def main() -> None:
    for idx, sample in enumerate(prompts, 1):
        t0 = time.time()
        plan = generate_control_plan(sample)
        print(f"[Prompt {idx}]")
        print(plan)
        print(f"[⏱ {time.time() - t0:.3f}s]\n")

if __name__ == "__main__":
    main()
