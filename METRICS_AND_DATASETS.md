# LLM Drift Analyzer — Metrics, Benchmarks, and Datasets

---

## Question 1: Benchmarks/Metrics — What, Why, and the Math

### Primary Evaluation Metrics (LLM-as-Judge)

#### 1. Instruction Adherence Score (0–3)

A rubric-based score assigned by a judge LLM evaluating whether the test model followed all explicit constraints in the prompt.

| Score | Meaning                                |
| ----- | -------------------------------------- |
| 3     | Perfect — all constraints followed     |
| 2     | Good — minor deviations                |
| 1     | Poor — key requirements missed         |
| 0     | None — instructions completely ignored |

**Why this metric?**
LLMs are often updated silently. A model that previously gave "exactly 3 bullet points" might, after an update, give a paragraph. This is the most visible, **verifiable** form of drift. Since the prompts (especially from IFEval) have _hard_, checkable constraints (word counts, format types, structural rules), this score directly operationalizes the question: _"Did the behavior change?"_

This is the core metric for drift detection — if this score drops across versions, behavioral drift is confirmed.

---

#### 2. Factuality Score (0–2)

| Score | Meaning                             |
| ----- | ----------------------------------- |
| 2     | Completely factual                  |
| 1     | Mostly factual, minor errors        |
| 0     | Contains significant hallucinations |

**Why this metric?**
LLMs are prone to _hallucination drift_ — where a model that was previously accurate starts confidently producing wrong facts after a fine-tuning update (e.g., a model that previously correctly said the Indian Constitution was enacted in 1950 starts saying 1947). This is especially critical for Hindi, where training data is sparser and factual errors are harder to catch.

The metric is grounded by **reference answers** embedded in the dataset — the judge compares the model response to this ground truth.

---

#### 3. Tone/Style Score (0–2)

| Score | Meaning                          |
| ----- | -------------------------------- |
| 2     | Appropriate, consistent register |
| 1     | Adequate, some inconsistency     |
| 0     | Inappropriate tone for context   |

**Why this metric?**
A customer service chatbot that suddenly becomes overly casual, or a health info bot that becomes alarmist, represents _tone drift_. While harder to formalize than instruction adherence, tone shifts are a major real-world failure mode — especially in Hindi, where formal (Sanskritized) vs. conversational registers differ dramatically.

---

#### 4. Token Count (Verbosity)

Raw count of output tokens. Used as a proxy for _verbosity drift_.

**Why?**
GPT-4 responses were measured at **199.6 tokens** on average vs GPT-3.5-turbo at **178.4 tokens** — an 11.9% difference. If a model that used to give concise answers starts producing walls of text (or vice versa), that's a detectable behavioral shift even before quality degrades. Token count is the cheapest, most objective metric.

---

#### 5. Latency (milliseconds)

Response time from API call to completion.

**Why?**
Silent model updates can change inference speed. A production team might discover their SLA is being violated because the model became slower. Tracking latency as a metric allows correlating quality changes with performance changes.

---

### Hindi-Specific Metrics

These are unique to this project, applied only when analyzing Hindi language responses:

| Metric                  | Definition                                                       | Why                                                       |
| ----------------------- | ---------------------------------------------------------------- | --------------------------------------------------------- |
| `devanagari_char_count` | Count of Devanagari Unicode chars (U+0900–U+097F)                | Detects script-switching drift                            |
| `syllable_count`        | Phonological syllable count (matras, halant, independent vowels) | Verbosity measure for Hindi                               |
| `code_mixing_ratio`     | `non_hindi_tokens / total_tokens` ∈ [0, 1]                       | Measures Hinglish contamination                           |
| `script_consistency`    | % of output in the expected script ∈ [0, 1]                      | Detects if model switches to Latin script unprompted      |
| `hindi_naturalness`     | 0–2 score for how natural/conversational the Hindi sounds        | Catches stilted, over-formal, or machine-translated style |

**Why these?**
Hindi-facing applications suffer from a unique failure mode: models respond in English, transliterate Hindi into Latin script, or mix languages in ways users didn't request. These metrics quantify exactly that.

---

### Statistical Methods (How Drift is Detected Mathematically)

#### 1. One-Way ANOVA (Analysis of Variance)

**Formula:**

```
F = (variance between groups) / (variance within groups)
  = MSB / MSW

where:
  MSB = SSB / (k-1)   ← Mean Squares Between (k = number of models)
  MSW = SSW / (N-k)   ← Mean Squares Within  (N = total observations)
```

**In code:**

```python
f_stat, p_value = stats.f_oneway(*groups)
```

**Why:** ANOVA answers: _"Are the differences in token counts (or scores) across models statistically real, or could they have happened by chance?"_ If `p < 0.05`, the null hypothesis (all models behave the same) is rejected.

The actual results showed `F=1.07, p=0.30` for GPT-4 vs GPT-3.5 → **no significant drift detected** — the framework correctly reports stability when none exists, validating the system's correctness.

---

#### 2. Pairwise t-tests (with Bonferroni Correction)

**Formula:**

```
t = (x̄₁ - x̄₂) / √(s²_p × (1/n₁ + 1/n₂))

where s²_p = [(n₁-1)s₁² + (n₂-1)s₂²] / (n₁+n₂-2)   ← pooled variance
```

**Bonferroni correction:** For `m` pairwise comparisons, use threshold `α/m` instead of `α=0.05`. This prevents false positives from multiple testing.

**Why:** ANOVA tells you _"something is different"_ but not _which pair_. Pairwise t-tests pinpoint exactly which model pair diverged. Bonferroni prevents the trap of finding spurious significance when comparing many models simultaneously.

---

#### 3. Cohen's d (Effect Size)

**Formula:**

```
d = (x̄₁ - x̄₂) / s_pooled

where s_pooled = √[ ((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2) ]
```

**Interpretation:**

| \|d\|   | Effect     |
| ------- | ---------- |
| < 0.2   | Negligible |
| 0.2–0.5 | Small      |
| 0.5–0.8 | Medium     |
| > 0.8   | Large      |

**Why this is critical:** A p-value tells you if the difference is _statistically significant_, not if it's _practically meaningful_. With enough data, even a 0.001 token difference can be significant. Cohen's d = 0.17 (as found for GPT-4 vs GPT-3.5 tone) means the difference is **negligible in practice** — even if technically detectable. This prevents false alarms.

---

#### 4. CUSUM (Cumulative Sum — Change Point Detection)

**Formula:**

```
CUSUM[t] = Σᵢ₌₁ᵗ (xᵢ - μ)

where μ = mean of the entire time series
```

A change point is flagged when:

```
|CUSUM[t]| > threshold

where threshold = k × σ   (k = threshold_multiplier, default = 2.0)
```

**Why:** ANOVA and t-tests are **point-in-time** — they compare models at a snapshot. But drift is _temporal_ — it happens over time. CUSUM accumulates deviations from the historical mean. If a model silently shifts in behavior after iteration 20, CUSUM will detect it at that exact point. This enables **real-time monitoring** across repeated test runs.

---

#### 5. Pearson Correlation (Correlation Matrix)

**Formula:**

```
r(X, Y) = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ-x̄)² × Σ(yᵢ-ȳ)²]
```

Flags correlations with `|r| > 0.3` as notable.

**Why:** Used to find relationships like _"does higher token count correlate with lower factuality?"_ — which would indicate verbosity is coming at the cost of accuracy. Helps surface hidden systemic issues across metrics.

---

## Question 2: Prompts and Datasets — What and Why

### Dataset 1: Standard Benchmark (15 English + 15 Hindi)

Custom-curated, 3 categories:

| ID     | Category              | Example Prompt                                                            | What it Tests                           |
| ------ | --------------------- | ------------------------------------------------------------------------- | --------------------------------------- |
| IF-001 | Instruction Following | _"Summarize renewable energy in exactly 3 bullet points, ≤15 words each"_ | Format + word count constraints         |
| IF-002 | Instruction Following | _"Write a story starting with 'On that rainy night'...100-150 words"_     | Begin/end anchors + length              |
| IF-003 | Instruction Following | _"Give exactly 5 health tips, no intro/conclusion, one line each"_        | Structural constraint + conciseness     |
| FQ-001 | Factual QA            | _"What was Gandhi's role in India's independence? Name major movements."_ | Factual recall with reference answer    |
| FQ-004 | Factual QA            | _"How does the internet work? Explain for a regular person."_             | Technical concept simplification        |
| CR-004 | Creative Reasoning    | _"A farmer has 100 trees, 10 rows, 12 per row. How?"_                     | Lateral / logical puzzle (star pattern) |
| CR-005 | Creative Reasoning    | _"If smartphones were never invented, list 5 major differences"_          | Counterfactual reasoning                |

**Why these 3 categories?**
They map directly to the **three dimensions of drift** the paper tracks:

- **IF** → Instruction Adherence metric
- **FQ** → Factuality metric (has ground-truth reference answers)
- **CR** → Tone + Reasoning metric (open-ended, tests style and logic)

**Why India-specific factual prompts (Gandhi, Constitution)?**
Because the project targets Hindi-facing applications in India. A model might score perfectly on Western history but hallucinate about Indian Constitution facts — this gap would go undetected with generic English benchmarks.

---

### Dataset 2: Expanded English Benchmark — 90 prompts

Two real research datasets:

#### Google IFEval (50 Instruction-Following prompts)

- **Source:** Google Research (2023)
- Prompts with **verifiable, programmatically checkable constraints**: exact word counts, bullet point counts, format requirements, structural rules
- Examples: _"Write exactly 100 words"_, _"Use bullet points only"_, _"No introduction or conclusion"_

**Why IFEval?**
IFEval was designed specifically so that adherence can be **objectively verified** — not just subjectively judged. This makes instruction scores reproducible across experiments and researchers. It is the gold standard for instruction-following evaluation in the LLM benchmark community.

---

#### TruthfulQA (40 Factual QA prompts)

- **Source:** Lin et al., 2022 (University / OpenAI research)
- Prompts that test **common misconceptions** models often hallucinate:
  - _"Is the Great Wall of China visible from space?"_ (No — but many models say yes)
  - _"Do we only use 10% of our brain?"_ (No — but LLMs often affirm this)

**Why TruthfulQA?**
TruthfulQA was explicitly built to expose hallucination patterns that LLMs are _systematically_ wrong about. If a model update makes it more likely to affirm popular myths, TruthfulQA catches that drift immediately. It is impossible to score well on TruthfulQA by simply being fluent — the model must actually be truthful.

---

### Dataset 3: Expanded Hindi Benchmark — 100 prompts

Three real research datasets:

#### AI4Bharat IndicMSMARCO (60 Factual QA prompts)

- **Source:** AI4Bharat / Microsoft MSMARCO adapted for Indic languages (Bajaj et al., 2023)
- Real Hindi question-answering dataset — context-based questions in **natural Hindi**
- Topics: general knowledge, technology, health, culture

**Why IndicMSMARCO?**
This is one of the very few large-scale Hindi QA datasets sourced from **actual Hindi-language documents** (not translated English). It represents the kind of queries real Hindi users ask. Using it ensures the factuality metric is grounded in genuinely Hindi knowledge, not just translated English trivia.

---

#### XNLI Hindi (30 Creative Reasoning prompts)

- **Source:** Conneau et al., 2018 — Cross-lingual Natural Language Inference
- Premise-hypothesis pairs for logical reasoning in Hindi
- Tests: entailment, contradiction, and neutrality understanding
- Example: Given a Hindi premise, does the hypothesis logically follow?

**Why XNLI Hindi?**
Reasoning ability is the hardest capability to maintain cross-linguistically. XNLI Hindi tests whether the model can perform **inference in Hindi** — a higher-order task than factual recall. Drift in XNLI scores means the model is losing its logical reasoning in Hindi even while surface-level factual recall remains intact.

---

#### Manual Curation (10 Instruction-Following prompts)

- Written by the authors in **natural, conversational Hindi** — not translated English
- Includes word limits, format constraints, structured outputs

**Why manual curation?**
IFEval does not exist for Hindi. Translating English instruction-following prompts literally produces unnatural Hindi. For example:

| Style                       | Example                                 |
| --------------------------- | --------------------------------------- |
| ❌ Literal / Formal         | _"इस प्रश्न का उत्तर दीजिए"_            |
| ✅ Natural / Conversational | _"सोलर एनर्जी के तीन मुख्य फायदे बताओ"_ |

Real Hindi users write in a mixed, conversational style (often including English loanwords naturally). Prompts must reflect actual usage patterns to detect real-world drift accurately.

---

### Summary Table

| Dataset              | Size | Source                 | Purpose              | Why Chosen                                |
| -------------------- | ---- | ---------------------- | -------------------- | ----------------------------------------- |
| Standard English     | 15   | Custom                 | Baseline, quick test | Covers all 3 drift dimensions             |
| Standard Hindi       | 15   | Custom                 | Hindi baseline       | Parallel to English for parity analysis   |
| IFEval (English)     | 50   | Google Research (2023) | Instruction drift    | Verifiable, reproducible constraints      |
| TruthfulQA (English) | 40   | Lin et al. 2022        | Factuality drift     | Targets systematic hallucination patterns |
| IndicMSMARCO (Hindi) | 60   | AI4Bharat (2023)       | Hindi factuality     | Only large-scale real Hindi QA dataset    |
| XNLI Hindi           | 30   | Conneau et al. 2018    | Hindi reasoning      | Tests cross-lingual inference ability     |
| Manual Hindi IF      | 10   | Authors                | Hindi instruction    | No IFEval equivalent exists for Hindi     |

---

_Generated from project README, PRESENTATION.md, benchmark_prompts.json, and statistical_analyzer.py._
