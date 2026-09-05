# Leakage audit: target `Logistics_Delay` (1000 rows, positive rate 0.566)

## Single conditions whose rows are pure in the target

| condition | rows | positives | pure label |
| --- | ---: | ---: | --- |
| `Shipment_Status == "Delayed"` | 350 | 350 | 1 |
| `Traffic_Status == "Heavy"` | 327 | 327 | 1 |

## Greedy OR-rule over pure-positive conditions

- rule: `Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"`
- accuracy 1.0 on all rows, 0 mismatches; positives covered 566/566

## Depth-2 decision tree, 5-fold CV

- mean accuracy 1.0 (min fold 1.0)

If a one-line rule or a depth-2 tree already reproduces the label, a larger model cannot add anything on this table.
