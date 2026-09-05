import csv
import json
import os

def csv_to_jsonl(csv_path: str, out_path: str = None):
    """
    将带表头的 CSV 转换为 JSONL，格式：
    {"conversations": [{"from": "system", "value": ...}, {"from": "user", "value": ...}, {"from": "assistant", "value": ...}]}
    - user: 使用除最后一列外的所有列，格式为 "列名: 值"（多行）
    - assistant: 只使用最后一列，格式为 "列名: 值"
    """
    system_text = (
        "Assume you are a supply chain analyst. Based on the following information, "
        "output the result for Logistics_Delay, where 1 represents a delay and 0 represents no delay."
    )

    # 输出路径默认与 CSV 同目录
    if out_path is None:
        out_path = os.path.join(os.path.dirname(csv_path), "all.jsonl")

    # 先尝试 utf-8-sig，再回退 gbk
    encodings_to_try = ["utf-8-sig", "gbk"]

    last_error = None
    for enc in encodings_to_try:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f_in, \
                 open(out_path, "w", encoding="utf-8", newline="\n") as f_out:

                reader = csv.DictReader(f_in)
                if not reader.fieldnames or len(reader.fieldnames) < 1:
                    raise ValueError("CSV 文件没有表头或列数为 0。")

                # 确定最后一列
                last_col = reader.fieldnames[-1]
                user_cols = reader.fieldnames[:-1]

                line_count = 0
                for row in reader:
                    # 生成 user 文本（前 N-1 列）
                    user_lines = []
                    for col in user_cols:
                        val = row.get(col, "")
                        # 统一转字符串并去除首尾空白
                        val_str = "" if val is None else str(val).strip()
                        user_lines.append(f"{col}: {val_str}")
                    user_text = "\n".join(user_lines)

                    # 生成 assistant 文本（仅最后一列）
                    last_val = row.get(last_col, "")
                    last_val_str = "" if last_val is None else str(last_val).strip()
                    assistant_text = f"{last_col}: {last_val_str}"

                    record = {
                        "conversations": [
                            {"from": "system", "value": system_text},
                            {"from": "user", "value": user_text},
                            {"from": "assistant", "value": assistant_text},
                        ]
                    }

                    # 按行写入 JSONL
                    f_out.write(json.dumps(record, ensure_ascii=False))
                    f_out.write("\n")
                    line_count += 1

                print(f"转换完成：{line_count} 行，已输出 -> {out_path}")
            # 成功读写则退出循环
            last_error = None
            break
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        # 两种编码都失败时抛出最后的异常
        raise last_error


if __name__ == "__main__":
    csv_path = r"C:\Users\yuchi\PycharmProjects\SC\data\smart_logistics_dataset.csv"
    # 默认输出到与 CSV 同目录的 all.jsonl
    csv_to_jsonl(csv_path)
