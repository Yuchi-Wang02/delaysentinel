# 一个 12.4 亿参数的模型学会了两个触发词

*我第一个微调项目的 label-leakage 事后复盘。文中所有指标都在
[`results/eval.json`](../results/eval.json) 或
[`results/olist_positive_control.json`](../results/olist_positive_control.json)；训练曲线的数字来自
`runs/*/trainer_state.json`；训练时长来自 `runs/sc904/tensorboard_events.json` 记录的 TensorBoard 事件文件
时间戳。图在 [`figures/`](figures/)。英文版见 `case_study.md`。*

## 1. 当时做了什么（2025 年 9 月）

我当时是会计与供应链管理双专业的本科生，想把 supervised fine-tuning 从头到尾走一遍。我拿了 Kaggle 上一张
1,000 行的 "Smart Logistics Supply Chain Dataset"，把每一行转成一条对话记录（15 行 `Column: value`，答案是
`Logistics_Delay: 0|1`），用没有设 seed 的 shuffle 切成 800/200，在一块 GPU（型号没有记录）上对
`Llama-3.2-1B-Instruct` 做了 30 个 epoch 的全参数 SFT，训练脚本改自一个开源智能家居项目。之后导出了 GGUF，
写了一个 Flask 表单，把权重传到 Hugging Face，model card 上写的是 "AI-powered logistics delay prediction"。

pipeline 是跑通的。训练 loss 的日志值在第 50 步、也就是第一个 epoch 结束时已四舍五入为 0.0000
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
| 两者皆是 | 111 | 111 |
| 两者皆否 | 434 | 0 |

`Logistics_Delay = 1 iff Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"` 在全部 1,000 行上成立，
零例外。第一个子句就是目标变量换了个列名；第二个是同一时刻快照里的路况：118 行已经 *Delivered* 却路况
*Heavy* 的记录全部被标为延误——任何"准时/延误"的业务定义都不会这样标。标签是对两列做的布尔运算。两列都被
原样写进了每一条训练 prompt。

现在 `python -m delaysentinel.leakage_audit` 会自动做这件事：扫描每一列，找出目标纯净的单条件，把纯正例
条件贪心地 OR 起来，报告两个条件就能零误差复现标签，并且深度 2 的决策树在 5 折交叉验证下是 1.000。训练前
跑一次，这个项目作为"预测"项目在第一天就该结束。

这张表也是合成的，不管 Kaggle 页面怎么说：坐标均匀铺满整个地球（`figures/fig_latlon.png`），每个数值列在
整数边界之间均匀分布，318 行没有延误的记录却带着"延误原因"。

## 4. Baseline

在同样的 200 行上，规则本身、两次分裂的决策树、logistic regression 和 gradient boosting 全是 1.000。全预测
为延误是 0.580（F1 0.734）。去掉两个规则列再训练 gradient boosting，accuracy 0.500、AUROC 0.452；再去掉事后
才有的 `Logistics_Delay_Reason` 是 0.475 / 0.460；加上从时间戳解析的月份、星期、小时是 0.520 / 0.482。表里
没有别的可学的东西。在全部 1,000 行上做 seeded 的重复 5 折交叉验证，每个 baseline 都是同样的结论（规则和
决策树每一折都是 1.000；三个"去掉规则列"的变体平均 accuracy 在 0.496 到 0.509 之间）。

所以这个微调模型不是"和决策树一样好"。在所有原始和反事实测试 prompt 上，它和一棵两次分裂、三个叶子的树
完全打平，代价是 1,235,814,400 个参数和约 18 分钟的训练时长（首末两个 TensorBoard 事件的间隔），而树几乎
不需要训练。两者只在树的 one-hot 编码器会映射为"未知"的输入上分道扬镳——下一节从这里开始。

## 5. 权重到底在做什么

分数只告诉你模型*匹配*了标签，不告诉你*怎么*匹配的。反事实探针逐个字段改写测试 prompt 的文本再重新打分，
并记录每次的 logit margin（`figures/fig_probes.png`）：

- 把 50 个"只因 Delayed 而为正"的样本的 `Shipment_Status` 改成 `In Transit`：50/50 翻成 0。
- 把 43 个"只因 Heavy 而为正"的样本的 `Traffic_Status` 改成 `Clear`：43/43 翻成 0。
- 把 84 个负例的 `Traffic_Status` 改成 `Heavy`，或把 `Shipment_Status` 改成 `Delayed`：各 84/84 翻成 1。
  合计 261 次规则字段编辑、177 个不同的行，全部按规则翻转。
- 每次改其余 13 个字段中的一个，对全部负例和全部正例做（3,200 条改写的 prompt）：0 个预测变化，
  每条的 margin 绝对值都在 12 以上。
- 从每条 prompt 里删掉两个规则行：模型对全部 200 行回答 0。

接着鲁棒性探针问这条规则是怎么被表示的，故事在这里变了。把两列改名、打乱 15 行的顺序，所有预测都不变，
所以列名无关紧要。只删掉一个规则行时，模型精确地执行另一条子句（只剩路况时 66 个正例，只剩状态时 73 个）。
到此为止它像一条绑定字段的规则。但是：

- 把 `Heavy` 写进 `Logistics_Delay_Reason`，或把 `Delayed` 写进 `Logistics_Delay_Reason` 或 `Asset_ID`，或把
  两个值在规则字段之间对调（`Shipment_Status: Heavy`、`Traffic_Status: Delayed`），对 84 个负例做：每种情况
  84 行全部输出 1。
- 把 `Delayed` 写成 `Not Delayed`，或把 `Heavy` 写成 `Not Heavy`：每一行仍然是 1。
- 把 `Delayed` 写成 `Late` 或 `Early`：73 行全是 1。写成 `Behind schedule`、`Postponed`、`Overdue`、`Held up`、
  `On Time` 或 `Pending`：只有同时路况 Heavy 的 23 行还是正例。
- 把 `Heavy` 写成 `Light`：66 行全是 1。写成 `Congested`、`Jammed`、`Gridlock`、`Slow`、`Dense`、
  `Free-flowing` 或 `Moderate`：只有同时状态 Delayed 的 23 行还是正例。

这组权重不是一条作用于两个字段的规则，而是一个对 `Delayed` 和 `Heavy` 两个 token 的检测器：出现在用户轮的
任何位置都触发，无视否定，不在乎它落在哪一列，在一个方向上比词义窄（每个子句五个同义词都不触发），在另一个
方向上比词义宽（`Early` 和 `Light` 会触发）。一棵 one-hot 决策树对这些改写全部会答 0；模型按训练数据从未要求
它学的词面相近度，有的答 1、有的答 0。

在 schema 之外模型不会拒答。我原来 model card 里那个用了 `carrier`、`weight_kg` 等模型从未见过的列的示例，
得到 `Logistics_Delay: 0`，margin 为 -14.0；只有列名的 prompt 是 -10.6；空 prompt 是 -2.1（整组里唯一一个
弱回答）。问它"法国的首都是哪里"，它回答 `Paris`。底模还在里面；微调加上的是一个狭窄的词面触发器，外加一个
"表单在但触发词不在时就答 0"的强先验。

## 6. 同一套方法在真实数据上得到什么

为了确认方法本身没问题，`python -m delaysentinel.positive_control` 用同样的规矩跑了公开的 Olist 巴西电商
订单数据（99,441 条真实、匿名化的订单，2016-2018）：客户收货日期晚于预计日期即为 late，只取已送达订单，
60 天的右删失保护，特征只用下单时可知的字段，按时间切分，测试期 37,702 条订单（prevalence 0.0722），报
bootstrap 区间。logistic regression 的 AUROC 是 0.7045 [0.6948, 0.7135]，AUPRC 0.1632 [0.1522, 0.1754]，
对比 prevalence 0.0722；histogram gradient boosting 是 0.6727 / 0.1299；只按承诺的送达周期排序是 0.5568
（`figures/fig_positive_control.png`）。对于"被标记就加急"这个决策，加急成本对每一个被标记的订单都要付，所以
阈值是成本比 `C_expedite / C_chargeback`；成本比 0.1 时 logistic 模型标记 7.4% 的订单，抓住 22.7% 的延误
订单，标记中 22.1% 是真延误。校准图显示模型在测试期系统性低估（最大的分箱里预测 0.034、实际 0.060），因为
训练窗口之后延误率上升了。这才是真实的下单时特征能给出的延误模型的样子：一个不大但可校准、带明确成本
权衡的信号，不是 100%。

## 7. 发布本身哪里错了

权重反而是问题最小的部分。

- **许可证。** 底模是 Llama 3.2，其社区许可证要求分发者附上协议、显示 "Built with Llama"、模型名以
  "Llama" 开头。我的仓库写的是 `apache-2.0`，没有许可证文件，名字叫 DelaySentinel。
- **署名。** 训练脚本是 `acon96/home-llm` 的 `train.py`（MIT，含 Stanford Alpaca 的 Apache-2.0 片段）的副本，
  没有任何致谢。上游后来删掉了这个文件，所以仓库现在钉住了它对应的 commit，并附上两份许可证全文。
- **model card。** 承诺了一个不存在的 Gradio Space，放了一个占位的仓库链接，示例代码用了错误的 repo id、
  自造的 prompt 格式和一个模型没见过的 schema，还把任务描述成"发货前"用"订单级特征"预测。整张卡没有一个
  指标。
- **配置文件。** `generation_config.json` 带着采样（底模默认值）发布；`config.json` 的 `use_cache: false`
  是训练遗留。采样恰好没有改变 200 个预测中的任何一个（三个 seed）——margin 都在 12 以上，这是必然的，
  但分类器不该采样。
- **应用。** Flask 界面从硬编码的本地路径加载模型，在推理时新增 pad token 并 resize embedding，为一位数
  的答案生成最多 512 个 token，用 `includes('1')` 解析结果，还让用户先填"延误原因"再预测延误。
- **评估。** 测试文件同时充当 Trainer 的 eval 集；由于上游 trainer 用 `SequentialSampler(Subset(...))`
  包了一层，每个 eval loss 点实际上都是在该文件的前 20 行上算的。没有定义 pad token 时，collator 还把
  `<|eot_id|>` 当作 pad 值，把 system 和 user 轮真正的结束符也遮蔽了。这两点在这个任务上没有观测到可测量
  的影响；在更难的任务上未做测试，而 eval 子集这个问题会让任何验证集悄悄缩成原来的 10%。

以上全部在 v1.0.0 里修掉了（`CHANGELOG.md`），没有一项需要动权重。Hub 那一侧由 `scripts/publish_hf.py`
在作者运行后推送。

## 8. 如果重来，按顺序

1. **先读许可证，再写 README。** 底模义务和数据条款放在最前面。
2. **先审计标签，再碰模型。** 把目标和每个分类列做交叉表；跑 leakage 扫描；拟合一棵深度 2 的树。
   有任何东西接近 1.000 就停下。
3. **确定每个字段在什么时点可知。** 对延误模型：下单、审批、交给承运商、送达。在结果发生时或之后才能
   观测到的都不是特征；*定义*了标签的字段在任何时点都不是特征。
4. **碰测试集之前冻结切分和协议。** seeded 切分、单独的验证期、写下 prevalence 和 trivial baseline、
   提前选定指标。
5. **每个数字旁边都放区间和 baseline。** accuracy 用 Wilson 或 Clopper-Pearson；有东西可重采样时用
   bootstrap；全预测为正的分数写在同一行。
6. **探针，而不只是打分。** 反事实改写很便宜，能告诉你模型在用什么；同义词和否定探针要在训练前设计，
   不是事后补。
7. **发布前署名和授权。** 借来的代码加文件头，正确的 license 标签，要求的声明文件，再加一道"card 不许引用
   任何文件里没有的数字"的检查。

## 9. 为什么这件事通向 BizHallu

教训不是"LLM 不擅长表格"，而是模型可以*因错误的原因而正确*，一个自信的答案不是理解的证据：一个 12.4 亿
参数的模型对 `Shipment_Status: Not Delayed` 回答 `1`，margin 和真正的 Delayed 一样大。这正是我现在的项目
[BizHallu](https://github.com/Yuchi-Wang02/bizhallu) 在 LLM 生成的零售分析里、按单个 business-fact span
研究的问题：每一条陈述是否有交易证据支撑，能不能被核查。我在那里用的习惯——带哈希的冻结切分、与每个指标
同行的 trivial baseline 和 bootstrap 区间、明确的 limitations、只引用入库 JSON 里数字的 card——都是从上面
这份修复清单开始的。

## 10. 复现

```bash
pip install -r requirements-lock.txt && pip install -e .
python -m pytest -q
python -m delaysentinel.eval --model Yuchiwang02/Llama-3.2-1B-DelaySentinel --out results/eval.json
python -m delaysentinel.positive_control --out results/olist_positive_control.json
python -m delaysentinel.eda --out docs/figures
python scripts/check_card_numbers.py
```
