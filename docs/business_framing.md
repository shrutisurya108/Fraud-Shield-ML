# Business Framing — Credit Card Fraud Detection
**fraud-shield-ml | Internal Document**
*Prepared for: Head of Risk Analytics*
*Phase: Problem Definition & Metric Selection*

---

## 1. Problem Definition

Credit card fraud is an **adversarial, rare-event classification problem**. Given a financial transaction and associated identity signals, the system must decide in near-real-time: *is this transaction fraudulent?*

Formally:

> Given feature vector **X** representing a transaction (amount, device, behavioural signals, merchant category, velocity features), predict binary label **y ∈ {0, 1}** where **1 = fraud**.

The dataset contains **590,540 transactions** spanning 6 months of real payment data provided by Vesta Corporation. The fraud rate is approximately **3.5%** — meaning the classes are severely imbalanced. A naive model that predicts "legitimate" for every transaction achieves 96.5% accuracy while catching **zero fraud**. This is why accuracy is not a usable metric here.

---

## 2. Cost Asymmetry — False Negatives vs False Positives

This is the most important business framing decision in any fraud detection system. The two error types have **fundamentally different costs**, and the model must be tuned to reflect this.

### False Negative (FN) — Missing real fraud
A fraudulent transaction is classified as legitimate and goes through.

**Costs incurred:**
- Full transaction value lost (charged back by card network to the merchant or issuing bank)
- Chargeback processing fee: typically $15–$100 per incident
- Regulatory reporting burden if volume exceeds thresholds
- Customer trust erosion — victims who experience undetected fraud churn at 3–5× the normal rate
- Reputational risk if fraud rates become public

**Severity: HIGH.** A single missed $5,000 fraudulent transaction costs far more than blocking 50 legitimate ones.

### False Positive (FP) — Blocking a legitimate transaction
A legitimate transaction is flagged as fraud and declined or sent for manual review.

**Costs incurred:**
- Friction and frustration for a genuine customer
- Potential abandoned purchase (lost revenue)
- Customer service call cost: ~$8–$15 per contact
- In extreme cases, customer churn

**Severity: MEDIUM.** Annoying and costly, but recoverable. The customer is still there. The card issuer can call to verify.

### Cost Ratio
Industry consensus across payments fraud literature places the FN:FP cost ratio between **5:1 and 20:1** depending on average transaction value and chargeback exposure. For this system we operate on a **conservative 5:1 assumption**, meaning we are willing to accept up to 5 false positives to avoid 1 false negative.

This asymmetry **directly drives our metric choice**.

---

## 3. Why Accuracy Fails Here

On this dataset, a model that predicts every transaction as legitimate scores:

```
Accuracy = 569,877 / 590,540 = 96.5%
```

This model is completely useless — it has 100% false negative rate. Accuracy cannot distinguish between a useless model and a good one when classes are imbalanced. It must never be used as the optimisation target or evaluation metric for this problem.

---

## 4. Metric Selection

### Primary Metric: F2 Score

The **F-beta score** generalises F1 by allowing us to weight recall and precision differently:

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

With **β = 2**, recall is weighted **twice as heavily** as precision. This reflects our cost asymmetry: missing fraud (low recall) is more damaging than incorrectly flagging legitimate transactions (low precision).

```
F2 = 5 × (Precision × Recall) / (4 × Precision + Recall)
```

F2 is our **primary optimisation target** for threshold selection and model comparison. All hyperparameter tuning decisions are made to maximise F2 on the validation set.

### Secondary Metric: PR-AUC (Average Precision)

**Precision-Recall AUC** measures the area under the Precision-Recall curve across all possible classification thresholds. It is threshold-independent and directly reflects performance on the minority class (fraud). PR-AUC is the **primary metric for comparing imbalance-handling strategies** and model architectures before threshold tuning.

ROC-AUC is also reported for completeness and benchmarking against published literature, but it is **not used for model selection** here — ROC-AUC is optimistic under class imbalance because it accounts for true negatives, which are trivially easy to get right when 96.5% of transactions are legitimate.

### Metric Summary

| Metric | Role | Why |
|--------|------|-----|
| **F2 Score** | Primary — model selection & threshold tuning | Weights recall 2× over precision, reflects FN > FP cost |
| **PR-AUC** | Primary — architecture & imbalance strategy comparison | Threshold-independent, minority-class focused |
| **Recall** | Reported — business-facing | "What % of fraud are we catching?" |
| **Precision** | Reported — operational | "Of flagged transactions, what % are real fraud?" |
| **ROC-AUC** | Reported — benchmarking only | Standard in literature; not used for decisions |
| **Accuracy** | Never used | Misleading under imbalance |

---

## 5. What "Good Enough" Looks Like

For a production fraud detection system at a mid-size payments company, industry benchmarks suggest:

- **Recall ≥ 0.80** — catching at least 80% of fraud is the baseline expectation
- **Precision ≥ 0.50** — fewer than half of flagged transactions should be false alarms at the operating threshold
- **F2 ≥ 0.70** — combined target reflecting the above
- **PR-AUC ≥ 0.75** — strong minority-class discrimination

These are portfolio targets. A top Kaggle submission on this dataset achieves PR-AUC ~0.92 with heavy feature engineering. Our goal is a well-engineered, explainable, production-grade system — not leaderboard optimisation.

---

## 6. Operational Context

The model output will be used in two modes:

**Hard block** — Transactions above a high-confidence fraud threshold are automatically declined. Requires very high precision to avoid customer friction.

**Review queue** — Transactions in a mid-confidence band are flagged for analyst review within 24 hours. Allows lower precision, higher recall.

Our threshold tuning in Phase 5 will optimise the primary threshold for F2, with a secondary high-precision threshold identified for the hard-block use case.

---

*This framing document is a living reference. All metric choices made in subsequent phases trace back to the cost asymmetry defined in Section 2.*
