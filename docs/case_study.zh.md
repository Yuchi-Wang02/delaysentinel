# 一个 12.4 亿参数的模型学会了照着 prompt 的表层作答

*我第一个微调项目的 label-leakage 事后复盘。文中所有指标都在
[`results/eval.json`](../results/eval.json) 或
[`results/olist_positive_control.json`](../results/olist_positive_control.json)；训练曲线的数字来自
`runs/*/trainer_state.json`；训练时长来自 `runs/sc904/tensorboard_events.json` 记录的 TensorBoard 事件文件
时间戳。图在 [`figures/`](figures/)。英文版见 `case_study.md`。*

## 1. 当时做了什么（2025 年 9 月）

我当时是会计与供应链管理双专业的本科生，想把 supervised fine-tuning 从头到尾走一遍。我拿了 Kaggle 上一张
1,000 行的 "Smart Logistics Supply Chain Dataset"，把每一行转成一条对话记录（15 行 `Column: value`，答案是
`Logistics_Delay: 0|1`），用没有设 seed 的 shuffle 切成 800/200，在一块 GPU（型号没有记录）上对
`Llama-3.2-1B-Instruct` 做了 30 个 epoch 的全参数 SFT，训练脚本取自一个开源智能家居项目。之后导出了 GGUF，
写了一个 Flask 表单，把权重传到 Hugging Face，model card 上写的是 "AI-powered logistics delay prediction"。

pipeline 是跑通的。训练 loss 的日志值在第 50 步、也就是第一个 epoch 结束时已四舍五入为 0.0000
（`figures/fig_training.png`）。我从来没有算过 accuracy，唯一看过的评估是 Trainer 的 eval loss，大约 3e-6。

## 2. 那个本该让我警觉的数字

一年后我第一次真正评估已发布的权重：在自己的 200 行测试集上用 greedy 解码，accuracy 1.000，F1 1.000，
混淆矩阵 `[[84, 0], [0, 116]]`，0 条无法解析。

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

这张表也是合成的，不管 Kaggle 页面怎么说：经纬度均匀铺满整个地球（`figures/fig_latlon.png`），318 行没有
延误的记录却带着"延误原因"。

## 4. Baseline

在同样的 200 行上，规则本身、两次分裂的决策树、logistic regression 和 gradient boosting 全是 1.000。全预测
为延误是 0.580（F1 0.734）。去掉两个规则列再训练 gradient boosting，accuracy 0.500、AUROC 0.452；再去掉事后
才有的 `Logistics_Delay_Reason` 是 0.475 / 0.460；加上从时间戳解析的月份、星期、小时是 0.520 / 0.482。表里
没有别的可学的东西。在全部 1,000 行上做 seeded 的重复 5 折交叉验证，每个 baseline 都是同样的结论（规则和
决策树每一折都是 1.000；三个"去掉规则列"的变体平均 accuracy 在 0.496 到 0.509 之间）。

所以这个微调模型不是"和决策树一样好"。在所有原始和反事实测试 prompt 上，它和一棵两次分裂、三个叶子的树
完全打平，代价是 1,235,814,400 个参数和约 18 分钟的训练日志时长，而树只需要 CPU 上的几毫秒。两者分道扬镳
的地方在下一节。

## 5. 权重到底在做什么

分数只告诉你模型*匹配*了标签，不告诉你*怎么*匹配的。反事实探针逐个字段改写测试 prompt 的文本再重新打分，
并记录每次的 logit margin（`figures/fig_probes.png`）：

- 把 50 个"只因 Delayed 而为正"的样本的 `Shipment_Status` 改成 `In Transit`：50/50 翻成 0。
- 把 43 个"只因 Heavy 而为正"的样本的 `Traffic_Status` 改成 `Clear`：43/43 翻成 0。
- 把 84 个负例的 `Traffic_Status` 改成 `Heavy`，或把 `Shipment_Status` 改成 `Delayed`：各 84/84 翻成 1。
  合计 261 次规则字段编辑、177 个不同的行，全部按规则翻转。
- 改其余 13 个字段中的一个（每行 15 次改写，因为 `Logistics_Delay_Reason` 用了三个替换值，共 3,000 次
  单字段编辑）或同时改两个（另 200 次）：0 个预测变化，每条的 margin
  绝对值都在 12 以上。这 3,200 次里有 197 次因为该行本来就是那个值而没有真正改动文本。
- 从每条 prompt 里删掉两个规则行：模型对全部 200 行回答 0。

到这里为止，一切都符合"模型学会了这条规则"。接着鲁棒性探针问：到底在匹配什么？答案不是规则。

**大小写、截断、否定都不影响。** `DELAYED`、`delayed`、只保留词干的 `Delay`、以及 `Not Delayed`，73 行全部
仍然是 1；`HEAVY`、`heavy`、截断的 `Heav`、`Not Heavy`，66 行全部仍然是 1。真学会了规则的模型对
`Not Delayed` 应该回答 0。

**词义基本不影响，少数影响的地方还指错了方向。** `Behind schedule`、`Postponed`、`Overdue`、`Held up`、
`On Time`、`Pending` 都*不*触发，只剩满足另一子句的 23 行；`Congested`、`Jammed`、`Gridlock`、`Slow`、
`Dense`、`Free-flowing`、`Moderate` 也一样。但 `Late`、`Early`、`Light` 每一行都触发。`Delayed` 的五个同义词
里有四个、`Heavy` 的五个同义词全部被读成"没延误"，而第五个同义词 `Late` 和两个反义词却被读成"延误"。

**字段也不影响。** 把 `Heavy` 或 `Delayed` 写进 `Logistics_Delay_Reason`，把 `Delayed` 写进 `Asset_ID`，
或者把两个值在规则字段之间对调，84 个负例全部输出 1。改列名、打乱 15 行的顺序，则完全没有变化。

**但这是一个特定的匹配，不是"见到陌生值就触发"。** 12 个与延误、路况无关的单 token 英文词（`Copper`、
`Violet`、`Harbor`、`Maple`、`Quartz`、`Falcon`、`Meadow`、`Cobalt`、`Lantern`、`Marble`、`Willow`、
`Amber`）分别放进两个规则字段，24 组探针里有 23 组维持基线不变。唯一的例外是 `Traffic_Status` 里的
`Meadow`，比基线多触发了 10 行。

**两个子句的实现方式不一样。** 给 84 个负例追加一行自由文本 "Note: the depot supervisor is Mr. Delayed"，
84 行全部变成 1；把同一行换成 "Mr. Heavy"，则毫无变化。`Delay` 的匹配扫描整个用户轮，`Heavy` 的匹配只读
字段值。

在 schema 之外模型不会拒答。我原来 model card 里那个用了 `carrier`、`weight_kg` 等模型从未见过的列的示例，
得到 `Logistics_Delay: 0`，margin 为 -14.0；只有列名的 prompt 是 -10.6；空 prompt 是 -2.1（四者中最弱）。
问它"法国的首都是哪里"，它回答 `Paris`。底模还在里面；微调加上的是一个只认表层写法的触发器，外加一个"表单在、
但里面没有任何东西触发时就答 0"的强先验。

仍然没有答案的是：这个模式到底是什么，而两个最顺手的猜测都已经被自己的数据排除。它不是对 `Delay`、`Heavy`
两个字符串的匹配 —— `Late`、`Early`、`Light` 一个都不含这两个子串，却每一行都触发。它也不是对整段 prompt 的
子串扫描 —— 字段名 `Logistics_Delay_Reason` 在全部 200 条 prompt 里都含有 "Delay"，却什么都不触发。为什么
偏偏是这一组写法触发、而 `Postponed`、`Congested` 不触发，没有测过。这些是我接下来会设计的探针，而它们本该
在训练前就写进协议，而不是一年以后。

## 6. 同一套指标在真实数据上是什么样

为了确认方法本身没问题，`python -m delaysentinel.positive_control` 把同一套规矩用在公开的 Olist 巴西电商
订单数据上（99,441 条真实、匿名化的订单，2016-2018）：客户收货日期晚于预计日期即为 late，只取已送达订单，
60 天的右删失保护，特征只用下单时可知的字段，切分方式则是一个真实部署能用的方式——训练集是 2018-03-01
之前*已送达*的订单（53,644 条，延误率 0.0505），测试集是该日期当天及之后*下单*的订单（37,702 条，延误率
0.0722）；跨越切分点的 3,673 条订单两边都不属于。

logistic regression 的 AUROC 是 0.6908，按订单重抽样的区间是 [0.6819, 0.6993]，按整月重抽样的区间是
[0.6329, 0.7599]；AUPRC 0.155，对比 prevalence 0.0722；histogram gradient boosting 是 0.6507 / 0.1176；
只按承诺的送达周期排序是 0.5568（`figures/fig_positive_control.png`）。按月重抽样的区间宽度约为按订单的七
倍，而"下一期还成不成立"这个问题对应的正是后者。

这个设置有两处问题，JSON 里如实写着而不是藏起来。第一，2,919 条在截止日前下单的订单在数据抽取时仍未送达，
且全部已过承诺日期；"只取已送达"的过滤器把它们丢掉，而不是记为 late。把其中落在测试窗口内的 555 条计为
late，测试集就从 37,702 条（2,722 条 late）变成 38,257 条（3,277 条 late），prevalence 从 0.0722 升到
0.0857，AUROC 变成 0.6812。第二，模型跨切分点失准：平均预测 0.0413，实际 0.0722。
这不是模型本可预见的基率漂移——测试窗口内的月度延误率从 2018 年 6 月的 0.0116 一直摆到 3 月的 0.1896，而模型
对那个 3 月的平均预测只有 0.0493。去掉 `purchase_month` 也修不好（同月 0.0565）。本次没有做任何重新校准。

把同一个 leakage 扫描器用在这张表上，找不到任何纯正例条件：贪心 OR 规则为空，深度 2 的决策树只有 0.9278，
就是多数类比例。这正是扫描器需要的阴性对照。

## 7. 发布本身哪里错了

权重反而是问题最小的部分。

- **许可证。** 底模是 Llama 3.2，其社区许可证要求分发者附上协议、显示 "Built with Llama"、模型名以
  "Llama" 开头。我的仓库写的是 `apache-2.0`，没有许可证文件，名字叫 DelaySentinel。
- **署名。** 训练脚本是 `acon96/home-llm` 的 `train.py`（MIT，含 Stanford Alpaca 的 Apache-2.0 片段）的
  副本，没有任何致谢。现在它原样、未经格式化地放在 `scripts/` 下，文件头写明对应的上游版本和我做的三处小改动。
- **model card。** 承诺了一个不存在的 Gradio Space，放了一个占位的仓库链接，示例代码用了错误的 repo id、
  自造的 prompt 格式和一个模型没见过的 schema，还把任务描述成"发货前"用"订单级特征"预测。整张卡没有一个指标。
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
在我运行之后推送。

## 8. 如果重来，按顺序

1. **先读许可证，再写 README。** 底模义务和数据条款放在最前面。
2. **先审计标签，再碰模型。** 把目标和每个分类列做交叉表；跑 leakage 扫描；拟合一棵深度 2 的树。
   有任何东西接近 1.000 就停下。
3. **确定每个字段在什么时点可知。** 对延误模型：下单、审批、交给承运商、送达。在结果发生时或之后才能
   观测到的都不是特征；*定义*了标签的字段在任何时点都不是特征。
4. **碰测试集之前冻结切分和协议。** seeded 切分、与测试期分开的验证期、写下 prevalence 和 trivial
   baseline、提前选定指标。
5. **每个数字旁边都放区间和 baseline**，而且要选与问题匹配的区间：问"这个估计有多精确"用按行 bootstrap，
   问"下一期还成不成立"用按月分块。
6. **探针，而不只是打分。** 反事实改写很便宜，能告诉你模型在用什么。同义词、否定和控制词探针要在训练前
   设计好；没有控制词的探针组无法区分"特定触发"和"见到陌生值就反应"。
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
