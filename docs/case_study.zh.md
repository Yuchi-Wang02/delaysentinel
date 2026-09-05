# 一个 12.4 亿参数的模型学会了一条两行的规则

*我第一个微调项目的 label-leakage 事后复盘。文中所有数字都在
[`results/eval.json`](../results/eval.json)，图在 [`figures/`](figures/)。英文版见 `case_study.md`。*

## 1. 当时做了什么（2025 年 9 月）

我是会计与供应链管理双专业的本科生，想把 supervised fine-tuning 从头到尾走一遍。我拿了 Kaggle 上一张
1,000 行的 "Smart Logistics Supply Chain Dataset"，把每一行转成一条对话记录（15 行 `Column: value`，
答案是 `Logistics_Delay: 0|1`），用没有设 seed 的 shuffle 切成 800/200，在消费级 GPU 上对
`Llama-3.2-1B-Instruct` 做了 30 个 epoch 的全参数 SFT，训练脚本改自一个开源智能家居项目。之后导出了
GGUF，写了一个 Flask 表单，把权重传到 Hugging Face，model card 上写的是 "AI-powered logistics delay
prediction"。

pipeline 是跑通的。训练 loss 在第 50 步、也就是第一个 epoch 结束时就精确为 0.0
（`figures/fig_training.png`）。我没有算过 accuracy，唯一看过的评估是 Trainer 的 eval loss，大约 3e-6。

## 2. 那个本该让我警觉的数字

一年后我用 greedy 解码在自己的 200 行测试集上重新评估已发布的权重：accuracy 1.000，F1 1.000，混淆矩阵
`[[84, 0], [0, 116]]`，0 条无法解析。

在一个有噪声的业务问题上拿到满分不是结果，是症状。第一件该查的事是：标签能不能被一个比模型简单得多的东西
从输入里直接复原。

## 3. 找到规则

把标签和两个"状态类"列做一张交叉表，一屏就能看完（`figures/fig_crosstab.png`）：

| 条件 | 行数 | 标记为延误 |
| --- | ---: | ---: |
| `Shipment_Status == "Delayed"` | 350 | 350 |
| `Traffic_Status == "Heavy"` | 327 | 327 |
| 两者皆否 | 434 | 0 |

`Logistics_Delay = 1 iff Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"` 在全部 1,000 行上
成立，零例外。第一个子句就是目标变量换了个列名；第二个是只有在货物已经在路上时才观测得到的路况。两列都被
原样写进了每一条训练 prompt。

现在 `python -m delaysentinel.leakage_audit` 会自动做这件事：扫描每一列，找出目标纯净的单条件，把纯正例
条件贪心地 OR 起来，报告两个条件就能零误差复现标签，并且深度 2 的决策树在 5 折交叉验证下是 1.000。它跑
一秒钟。训练前跑一次，这个项目作为"预测"项目在第一天就该结束。

这张表也是合成的，不管 Kaggle 页面怎么说：坐标均匀铺满整个地球（`figures/fig_latlon.png`），每个数值列
在整数边界之间均匀分布，318 行没有延误的记录却带着"延误原因"。

## 4. Baseline

在同样的 200 行上，规则本身、深度 2 决策树、logistic regression 和 gradient boosting 全是 1.000。全预测为
延误是 0.580（F1 0.734）。去掉两个规则列再训练 gradient boosting，accuracy 0.500、AUROC 0.452：表里没有
别的可学的东西。在全部 1,000 行上做 seeded 的重复 5 折交叉验证，每个 baseline 都是同样的结论（规则和
决策树每一折都是 1.000；去掉规则列的 boosting 是 0.496）。

所以这个微调模型不是"和决策树一样好"，而是和一棵三个叶子的决策树*无法区分*，参数量大约多了十亿倍，
GPU 时间从零变成了约 18 分钟。

## 5. 权重到底在做什么

分数只告诉你模型*匹配*了标签，不告诉你*怎么*匹配的。反事实探针逐个字段改写测试 prompt 的文本再重新打分
（`figures/fig_probes.png`）：

- 把 50 个"只因 Delayed 而为正"的样本的 `Shipment_Status` 改成 `In Transit`：50/50 翻成 0。
- 把 43 个"只因 Heavy 而为正"的样本的 `Traffic_Status` 改成 `Clear`：43/43 翻成 0。
- 把 84 个负例的 `Traffic_Status` 改成 `Heavy`，或把 `Shipment_Status` 改成 `Delayed`：各 84/84 翻成 1。
- 把负例的 `Waiting_Time` 改成 60、`Temperature` 改成 30，或把正例改成 10 和 18：200 个预测 0 个变化。
- 从每条 prompt 里删掉两个规则行：模型对全部 200 行回答 0。

鲁棒性探针进一步问这条规则是怎么被表示的。把两列改名、只改一列、打乱 15 行的顺序，所有预测都不变，
所以它不是对列名的字面查表。把 `Delayed` 写成 `Late`、`DELAYED` 或 `delayed`，73 行仍然全是 1；
`HEAVY`、`heavy` 也一样，66 行全是 1。但 `Congested` 不被当成 `Heavy`：只有同时是 `Delayed` 的 23 行
还是正例。只删掉一个规则行时，模型精确地执行另一条子句（只剩路况时 66 个正例，只剩状态时 73 个，与剩余
子句一致率都是 1.000）。没见过的取值（`Unknown`、`N/A`）一律输出 0。

在 schema 之外模型不会拒答。我原来 model card 里那个用了 `carrier`、`weight_kg` 等模型从未见过的列的
示例，得到一个自信的 `Logistics_Delay: 0`；空 prompt、只有列名的 prompt 也一样。问它"法国的首都是哪里"，
它回答 `Paris`。底模还在里面；微调加上的是一个狭窄、字面的两个字符串相等的 `OR`，外加一个"表单在但触发
值不在时就答 0"的强先验。

## 6. 发布本身哪里错了

权重反而是问题最小的部分。

- **许可证。** 底模是 Llama 3.2，其社区许可证要求分发者附上协议、显示 "Built with Llama"、模型名以
  "Llama" 开头。我的仓库写的是 `apache-2.0`，没有许可证文件，名字叫 DelaySentinel。
- **署名。** 训练脚本是 `acon96/home-llm` 的 `train.py`（MIT）的副本，没有任何致谢。上游后来删掉了
  这个文件，所以仓库现在钉住了它对应的 commit。
- **model card。** 承诺了一个不存在的 Gradio Space，放了一个占位的仓库链接，示例代码用了错误的 repo id、
  自造的 prompt 格式和一个模型没见过的 schema，还把任务描述成"发货前"用"订单级特征"预测。整张卡没有一个
  指标。
- **配置文件。** `generation_config.json` 带着采样（底模默认值）发布；`config.json` 的 `use_cache: false`
  是训练遗留。采样恰好没有改变 200 个预测中的任何一个（三个 seed），但分类器不该采样。
- **应用。** Flask 界面从硬编码的本地路径加载模型，在推理时新增 pad token 并 resize embedding，为一位数
  的答案生成最多 512 个 token，用 `includes('1')` 解析结果，还让用户先填"延误原因"再预测延误。
- **评估。** 测试文件同时充当 Trainer 的 eval 集；由于上游 trainer 用 `SequentialSampler(Subset(...))`
  包了一层，每个 eval loss 点实际上都是在该文件的前 20 行上算的。没有定义 pad token 时，collator 还把
  `<|eot_id|>` 当作 pad 值，把 system 和 user 轮真正的结束符也遮蔽了。这两点对这个任务没影响，对真实任务
  都会有。

以上全部在 v1.0.0 里修掉了（`CHANGELOG.md`），没有一项需要动权重。

## 7. 如果重来，按顺序

1. **先读许可证，再写 README。** 底模义务和数据条款放在最前面。
2. **先审计标签，再碰模型。** 把目标和每个分类列做交叉表；跑 leakage 扫描；拟合一棵深度 2 的树。
   有任何东西接近 1.000 就停下。
3. **确定每个字段在什么时点可知。** 对延误模型：下单、审批、交给承运商、送达。在结果发生时或之后才能
   观测到的都不是特征。
4. **碰测试集之前冻结切分和协议。** seeded 切分、单独的验证集、写下 prevalence 和 trivial baseline、
   提前选定指标。
5. **每个数字旁边都放区间和 baseline。** accuracy 用 Wilson 或 Clopper-Pearson；有东西可重采样时用
   bootstrap；全预测为正的分数写在同一行。
6. **探针，而不只是打分。** 反事实改写很便宜，能告诉你模型在用什么。
7. **发布前署名和授权。** 借来的代码加文件头，正确的 license 标签，要求的声明文件。

## 8. 为什么这件事通向 BizHallu

教训不是"LLM 不擅长表格"，而是模型可以*因错误的原因而正确*，而一个自信的答案不是理解的证据。这正是我
现在的项目 [BizHallu](https://github.com/Yuchi-Wang02/bizhallu) 在 LLM 生成的零售分析里、按单个
business-fact span 研究的问题：每一条陈述是否有交易证据支撑，能不能被核查。我在那里用的习惯——带哈希的
冻结切分、与每个指标同行的 trivial baseline 和 bootstrap 区间、明确的 limitations、只引用入库 JSON 里
数字的 card——都是从上面这份修复清单开始的。

## 9. 复现

```bash
pip install -e ".[model,figures,dev]"
python -m pytest -q
python -m delaysentinel.eval --model Yuchiwang02/Llama-3.2-1B-DelaySentinel --out results/eval.json
python -m delaysentinel.eda --out docs/figures
python scripts/check_card_numbers.py
```
