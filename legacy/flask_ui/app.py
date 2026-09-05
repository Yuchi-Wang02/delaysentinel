#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
app.py - Flask web application for LLAMA model visualization
"""

import os
import json
import time
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# ─────────────────── 基本配置 ───────────────────
MODEL_DIR = "models/VCU-test"  # ← 如有需要请自行修改
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.environ.get("HF_TOKEN")  # 在环境变量里放入 Hugging Face 令牌

# ─────────────────── 全局变量存储模型和分词器 ───────────────────
model = None
tokenizer = None


def load_model():
    """加载模型和分词器"""
    global model, tokenizer

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_TOKEN)

    # LLAMA 默认没有 <pad>，我们补一个，避免 "pad 与 eos 相同" 警告
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, token=HF_TOKEN)

    # 调整词嵌入大小
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    model.to(DEVICE)
    print(f"Model loaded successfully on {DEVICE}")


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
        add_generation_prompt=True  # 在末尾自动补一个 <assistant>
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


# ─────────────────── Flask 路由 ───────────────────
@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.json

        # 构建系统提示
        system_prompt = "Assume you are a supply chain analyst. Based on the following information, output the result for Logistics_Delay, where 1 represents a delay and 0 represents no delay."

        # 构建用户输入
        user_content_parts = []

        # 添加所有非空字段
        fields = [
            ('Timestamp', data.get('timestamp')),
            ('Asset_ID', data.get('asset_id')),
            ('Latitude', data.get('latitude')),
            ('Longitude', data.get('longitude')),
            ('Inventory_Level', data.get('inventory_level')),
            ('Shipment_Status', data.get('shipment_status')),
            ('Temperature', data.get('temperature')),
            ('Humidity', data.get('humidity')),
            ('Traffic_Status', data.get('traffic_status')),
            ('Waiting_Time', data.get('waiting_time')),
            ('User_Transaction_Amount', data.get('transaction_amount')),
            ('User_Purchase_Frequency', data.get('purchase_frequency')),
            ('Logistics_Delay_Reason', data.get('delay_reason')),
            ('Asset_Utilization', data.get('asset_utilization')),
            ('Demand_Forecast', data.get('demand_forecast'))
        ]

        for field_name, field_value in fields:
            if field_value and str(field_value).strip():
                user_content_parts.append(f"{field_name}: {field_value}")

        user_content = "\n".join(user_content_parts)

        # 构建对话格式
        dialog = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "user", "value": user_content}
            ]
        }

        # 生成预测
        start_time = time.time()
        result = generate_control_plan(dialog)
        inference_time = time.time() - start_time

        return jsonify({
            'success': True,
            'result': result,
            'inference_time': f"{inference_time:.3f}s",
            'input_dialog': dialog
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })


# ─────────────────── 主程序 ───────────────────
if __name__ == '__main__':
    # 启动时加载模型
    load_model()

    # 启动 Flask 应用
    app.run(debug=False, host='0.0.0.0', port=5000)