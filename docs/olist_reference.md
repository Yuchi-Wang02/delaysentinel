# A separate study on real Olist orders

This study applies classical models to a defined delivery outcome with features intended to be
available at checkout and a later test period. It extends the evaluation workflow to a different
dataset. **The DelaySentinel Llama checkpoint was not trained or evaluated on Olist.**

[Project overview](https://github.com/Yuchi-Wang02/delaysentinel) ·
[Aggregate results](../results/olist_positive_control.json) ·
[Implementation](https://github.com/Yuchi-Wang02/delaysentinel/blob/main/src/delaysentinel/positive_control.py) ·
[Reproduction commands](reproduce.md#separate-olist-study)

## Data and task

`python -m delaysentinel.positive_control` runs logistic regression and histogram
gradient boosting on the public Olist Brazilian e-commerce
orders (99,441 real, anonymised orders, 2016-2018; Kaggle licence CC BY-NC-SA 4.0, used
non-commercially with attribution; no Olist row is committed here). Label: `late =
delivered_customer_date > estimated_delivery_date`, comparing calendar dates on delivered orders. Features are selected to represent
information available at checkout; their operational availability is an assumption of this study. Numbers in
[`results/olist_positive_control.json`](../results/olist_positive_control.json).

## Cohort and temporal split

Three filters run before the model sees anything, and the JSON reports each. The 60-day
right-censoring guard keeps only orders purchased on or before 2018-08-18, taking 99,441 orders
down to 97,938. Of those, 2,919 were still undelivered when the data was extracted, every one of
them already past its promised date: 1,723 in flight, 1,188 cancelled or unavailable, and 8 marked
`delivered` with no delivery timestamp. The delivered-only filter removes all 2,919 instead of
labelling them late, which is a survivorship filter, not a censoring correction; the sensitivity
row below adds back the 555 in-flight orders that fall in the test window. That leaves 95,019 orders to
model. Separately, a model
deployed on 2018-03-01 can only train on labels that exist by then, so the training set is orders
*delivered* before that date (53,644 orders, late rate 0.0505) and the test set is orders
*purchased* on or after it (37,702 orders, late rate 0.0722); 3,673 orders purchased before the
split but delivered after it, 1,075 of them late, belong to neither.

## Recorded results

The brackets report the stored bootstrap intervals. AUROC describes ranking discrimination;
AUPRC should be compared with the test positive rate, and Brier measures probability error.

| model (test period) | AUROC [orders] | AUROC [months] | AUPRC | Brier | mean predicted |
| --- | --- | --- | --- | ---: | ---: |
| constant = training late rate | 0.500 | n/a | 0.0722 (= prevalence) | 0.0675 | 0.0505 |
| promised lead time only (shorter = riskier) | 0.5568 | n/a | 0.0953 | n/a | n/a |
| logistic regression | 0.6908 [0.6819, 0.6993] | [0.6329, 0.7599] | 0.155 [0.1445, 0.1669] | 0.0657 | 0.0413 |
| … without `purchase_month` | 0.6829 [0.6729, 0.6922] | [0.6355, 0.7472] | 0.139 [0.1305, 0.149] | 0.0654 | 0.0558 |
| histogram gradient boosting | 0.6507 [0.6397, 0.6608] | [0.607, 0.7502] | 0.1176 [0.111, 0.1248] | 0.0668 | 0.0504 |
| logistic regression, in-flight orders counted late | 0.6812 [0.6718, 0.6896] | [0.626, 0.7437] | 0.1742 [0.165, 0.1844] | 0.0777 | 0.0415 (prevalence 0.0857) |

## Variation and calibration

The order-level bootstrap resamples orders within the observed test period. The month-block
bootstrap resamples observed months and produces a substantially wider interval here. It
reflects sensitivity to the mix of observed months; it does not guarantee performance in a
future period or account for every form of temporal dependence.

The test late rate ranges from 0.0116 in June 2018 to 0.1896 in March 2018. The logistic model
predicts an average 0.0493 for March; dropping `purchase_month` changes that to 0.0565.
Across the whole test set, mean predicted probability is 0.0413 against an observed rate of
0.0722. This is evidence of calibration mismatch in the observed period. No recalibration
was attempted, and a prospective evaluation would need a separate validation period.

## Connecting scores to a business decision

For the decision "expedite if flagged", with an expedite cost paid on every flagged order and a
late-delivery cost avoided with efficacy *e*, under this simplified cost model the threshold on a *calibrated* probability
is `p* = C_expedite / (e · C_late)`. The sweep in the JSON is therefore reported as score
thresholds, not cost ratios: at a score of 0.1 the logistic model flags 6.2% of orders and catches
18.9% of late ones, and 21.9% of the flagged orders really are late.

## Leakage-scanner comparison

The same leakage scanner used on DelaySentinel finds **no**
pure-positive condition here: the greedy OR-rule is empty, and a depth-2 tree reaches 0.9278,
which is just the majority class. Three pure-negative conditions appear, each covering 20 to 26 rows. Such small cells do not
establish generalizable rules. This comparison supplies a negative control for the particular
scanner; it is not a certificate that the dataset contains no leakage.

![Olist reference study](figures/fig_positive_control.png)

## Features and unresolved assumptions

Features include purchase month, weekday and time-of-day bin; promised lead time; customer and
seller state; first-item category; item count; total price and freight; payment type and
installments; and distance between customer and seller ZIP-code centroids. Imputation and
encoding are fitted in the training pipeline.

The recorded order table approximates checkout-time information. A deployment would require
source-system confirmation of when each value and promised date becomes available. It would
also need an explicit treatment of cancellations, unfinished deliveries, multi-seller orders,
geographic drift, and the action triggered by a risk score.

The delivered-only filter excludes unfinished orders; the sensitivity result adds the
in-flight orders from the test period that were already beyond their promised date. Those
definitions answer related but different questions. The sample crossing the train/test
boundary is excluded because its label would not yet exist at the proposed training date.

There is no separate validation period for choosing calibration or decision thresholds, no
prospective deployment test, and no evaluation of intervention effectiveness. The threshold
sweep is descriptive; its results are not measured savings or a proven dispatch policy.

## Source and license

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
used for non-commercial research and teaching under CC BY-NC-SA 4.0. Runtime downloads use
the public `aviahYadler/Olist_Ecommerce_Dataset` mirror, which does not provide its own license
statement. File hashes are recorded in `provenance.file_sha256` in the aggregate result JSON.

No Olist rows are committed to this repository. The aggregate result file is distributed under
CC BY-NC-SA 4.0; implementation code is MIT. See [attribution and licenses](../THIRD_PARTY_LICENSES.md).
