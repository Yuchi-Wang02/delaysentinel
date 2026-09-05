import json
import random
import os

def split_jsonl(file_path: str, train_ratio=0.8):
    # 默认输出路径与输入文件同目录
    base_dir = os.path.dirname(file_path)
    train_path = os.path.join(base_dir, "train.jsonl")
    test_path = os.path.join(base_dir, "test.jsonl")

    # 读取所有行
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"原始样本数: {len(lines)}")

    # 打乱
    random.shuffle(lines)

    # 分割
    split_index = int(len(lines) * train_ratio)
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]

    # 写入训练集
    with open(train_path, "w", encoding="utf-8") as f_train:
        for line in train_lines:
            f_train.write(line)

    # 写入测试集
    with open(test_path, "w", encoding="utf-8") as f_test:
        for line in test_lines:
            f_test.write(line)

    print(f"训练集: {len(train_lines)} 行 -> {train_path}")
    print(f"测试集: {len(test_lines)} 行 -> {test_path}")


if __name__ == "__main__":
    file_path = r"C:\Users\yuchi\PycharmProjects\SC\data\all.jsonl"
    split_jsonl(file_path)
