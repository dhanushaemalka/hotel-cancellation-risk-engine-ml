# IT3091 Machine Learning — Initial Project Proposal & Framing Brief
**Group Assignment Track**: Guided Data Track  
**Assigned Group Code**: 6  
**Domain**: Tourism & Hospitality  
**Dataset**: Hotel Booking Demand (Antonio, de Almeida & Nunes, 2019)  
**Target Submission Date**: Semester Project Milestone 1  

---

## 1. Track & Problem Identification
- **Track**: Guided Data Track (Controlled Domain: Tourism & Hospitality).
- **Business Stakeholders**: Hotel General Managers (GM), Revenue Management Directors, Reservations & Front Desk Operations Directorate.
- **Business Scenario**: A mid-to-upscale hotel chain operating both resort and city properties suffers severe revenue unpredictability and operational friction due to an aggregate booking cancellation rate exceeding **37.0%**. Cancellations lead to vacant, perishable room inventory (underbooking revenue loss), while unguided overbooking causes severe customer walking penalties and brand damage.
- **Unit of Analysis**: An individual confirmed reservation record ($N = 119,390$).

---

## 2. Decision Lenses Selection & Defensible Justification

### Primary Decision Lens: Cancellation Risk Prediction & Dynamic Revenue Protection
- **Paradigm**: Supervised Binary Classification & Cost-Sensitive Threshold Optimization.
- **Exact Task**: Predict the posterior probability $P(\text{is\_canceled} = 1 \mid \mathbf{x}_i)$ for any incoming reservation $\mathbf{x}_i$ at booking confirmation time.
- **Expected Output**: 
  1. Calibrated cancellation probability ($0.0 \dots 1.0$).
  2. 3-tier operational risk classification (`Low Risk` $<25\%$, `Medium Risk` $25-55\%$, `High Risk` $>55\%$).
  3. Net Revenue-maximizing threshold $T^*$ derived from empirical cost matrix ($C_{FN} = \$150$ room loss vs $C_{FP} = \$40$ overbooking friction).
- **Justification**: Hotel rooms are strictly perishable assets; an unsold room night at 00:00 produces zero residual value. Predicting cancellation probability enables proactive inventory resale, automated overbooking buffer setting, and targeted deposit requirements.

### Optional Secondary Decision Lens: Booking & Customer Behavioral Segmentation
- **Paradigm**: Unsupervised Machine Learning ($K$-Means Clustering + Silhouette / Elbow Optimization).
- **Exact Task**: Group historical reservations into distinct behavioral personas based on lead time, length of stay, party composition, average daily rate (ADR), cancellation history, and distribution channel.
- **Expected Output**: 4 distinct, interpretable guest archetypes with customized policy interventions.
- **Coherence Justification**: Directly strengthens the primary lens because cancellation risk drivers differ fundamentally across guest types: corporate transient guests cancel due to schedule changes, while long-lead vacation groups cancel due to trip replanning. Applying uniform cancellation policies harms high-value loyal guests; segment-tailored policies optimize both customer retention and risk control.

---

## 3. High-Level Workflow Architecture

```
[1. Business Framing] 
        │
        ▼
[2. Data Ingestion & Quality Auditing] (32 attributes, 119k records)
        │
        ▼
[3. Leakage Elimination & Preprocessing] (Drop reservation_status, Impute, Encode, Scale)
        │
        ├──► [4A. Supervised Risk Modeling] (Logistic, RF, XGBoost, LightGBM, MLP)
        │          │
        │          ▼
        │    [5A. Cost-Benefit & Threshold Optimization] (T* for Max Net Profit)
        │          │
        │          ▼
        │    [6A. Explainable AI (SHAP)] (Beeswarm, Feature Rankings, Waterfall)
        │
        └──► [4B. Unsupervised Segmentation] (K-Means on Behavioral Features)
                   │
                   ▼
             [5B. Persona Profiling & Policy Mapping]
                   │
                   ▼
[7. Operational Decision Support System & Executive Web Dashboard]
```

---

## 4. Core Team Member Responsibilities Matrix

| Team Member Name | Student ID | Core Functional Responsibility | Key Deliverables & Evidence |
|---|---|---|---|
| **Member 1 (Lead / ML Engineer)** | IT3091-M1 | Pipeline Architecture, Leakage Prevention & Advanced Modeling | `src/preprocessing.py`, `src/models.py`, 5-Fold Stratified CV, XGBoost/LightGBM tuning |
| **Member 2 (Data Analyst & EDA)** | IT3091-M2 | Data Profiling, Data Dictionary, Statistical Hypotheses & Visualizations | `reports/DATA_DICTIONARY.md`, `src/eda_engine.py`, 6 EDA figures, EDA Insight Log |
| **Member 3 (AI Evaluation & Economics)** | IT3091-M3 | Multi-Metric Evaluation, Cost-Benefit Optimization & SHAP Explainability | `src/evaluation.py`, `src/explainability.py`, ROC/PR curves, Expected Value Matrix, SHAP plots |
| **Member 4 (Decision Support & UX)** | IT3091-M4 | Unsupervised Segmentation, Decision Engine & Interactive Dashboard | `src/segmentation.py`, `src/decision_support.py`, `dashboard/`, Final Presentation Video |

---

## 5. Preliminary Timeline & Milestone Schedule
- **Week 1-2**: Problem framing, dataset auditing, data dictionary, initial proposal submission.
- **Week 3-4**: Leakage-free preprocessing pipeline, baseline models, EDA insight log generation.
- **Week 5-6**: Advanced model benchmarking (RF, XGBoost, LightGBM, MLP), SHAP explainability, K-Means segmentation, Cost-Benefit optimization.
- **Week 7**: Executive dashboard integration, Final Report with all 6 evidence logs, 3-minute YouTube demo recording, individual learning journey reflections.
