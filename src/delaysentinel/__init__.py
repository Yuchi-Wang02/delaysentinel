"""DelaySentinel: a label-leakage case study around a Llama-3.2-1B-Instruct fine-tune.

The package contains everything needed to reproduce the numbers in the model card:

- :mod:`delaysentinel.prompting`   - the exact prompt format used in training, label parsing, prompt edits
- :mod:`delaysentinel.data`        - loading the frozen split, the two-clause label rule, hashing
- :mod:`delaysentinel.stats`       - metrics with Wilson / Clopper-Pearson / bootstrap intervals
- :mod:`delaysentinel.baselines`   - rule, tree, logistic and boosting baselines on the same split
- :mod:`delaysentinel.model`       - inference wrapper around the published weights (needs torch)
- :mod:`delaysentinel.probes`      - counterfactual, robustness and out-of-distribution probes
- :mod:`delaysentinel.eval`        - the command-line entry point that writes ``results/eval.json``
- :mod:`delaysentinel.eda`         - the figures used in the case study
- :mod:`delaysentinel.leakage_audit` - a generic "can a one-line rule reproduce the label?" scanner
"""

__version__ = "1.0.2"
