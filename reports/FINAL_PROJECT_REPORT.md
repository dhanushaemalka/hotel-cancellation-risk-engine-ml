# IT3091 Machine Learning — Final Group Project Report
# Hotel Booking Demand Intelligence & Revenue-Loss Mitigation System
**Domain 6**: Tourism & Hospitality  
**Dataset**: Hotel Booking Demand (Antonio, de Almeida & Nunes, 2019)  
**Group Code**: 6 | **Submission Track**: Guided Data Track  
**Date of Evaluation**: Academic Year 2026/2027  

---

## Executive Summary
In the hospitality industry, hotel rooms represent strictly **perishable inventory**; an unoccupied room night at midnight produces zero residual value, permanently destroying potential revenue. Across the 119,390 historical reservations in our benchmark dataset, cancellations account for a staggering **37.08%** of all bookings (44,224 lost reservations). 

To solve this operational and financial challenge, this project develops a dual-lens Machine Learning and Decision Support framework:
1. **Primary Decision Lens (Supervised Classification & Cost-Sensitive Optimization)**: Predicts the individual cancellation probability $P(\text{is\_canceled}=1 \mid \mathbf{x}_i)$ at the time of reservation confirmation, categorizes reservations into operational risk tiers (`Low`, `Medium`, `High`), and applies an empirical **Expected Value Cost Matrix** ($C_{FN}=\$150$ room loss vs $C_{FP}=\$40$ overbooking friction) to identify the profit-maximizing operational decision threshold **$T^* = 0.39$**, delivering an annualized net profit lift of **+$84,556.66 per 100,000 bookings**.
2. **Secondary Decision Lens (Unsupervised $K$-Means Segmentation)**: Clusters guests into 4 distinct behavioral archetypes (*Spontaneous Business*, *High-Risk Long-Lead*, *Premium Family*, and *Loyal Repeat*), enabling tailored deposit terms and personalized pre-arrival retention workflows.

Benchmarking seven model architectures under strict **5-Fold Stratified Cross-Validation** demonstrates that gradient boosted trees (**XGBoost / LightGBM**) and tree ensembles (**Random Forest**) achieve superior performance (**Test ROC-AUC: 0.9473**, **PR-AUC: 0.9239**, **F1-Score: 0.8317**, **Brier Calibration Score: 0.0926**), massively outperforming heuristic and linear baselines. Global and local explainability via **SHAP (SHapley Additive exPlanations)** uncovers the non-linear mechanisms driving cancellation risk, notably resolving the empirical **"Deposit Type Paradox"** and highlighting customer commitment signals.

---

## 1. Problem Framing & Dual Decision Lenses Formulation (5 Marks)

### 1.1 Problem Framing Canvas
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              PROBLEM FRAMING CANVAS                                     │
├──────────────────────────────┬──────────────────────────────────────────────────────────┤
│ 1. Primary Stakeholders      │ • Hotel Revenue Management Directorate                   │
│                              │ • Front Desk & Reservations Operations Directorate       │
│                              │ • Hotel Property General Managers (H1 Resort, H2 City)   │
├──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ 2. Core Business Dilemma     │ • Underbooking Risk: Perishable room vacancy (~37% lost) │
│                              │ • Overbooking Risk: Relocation/walk penalties ($250+)    │
│                              │ • One-Size-Fits-All Policy: Harms loyal, low-risk guests │
├──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ 3. Unit of Analysis          │ An individual confirmed reservation record (N = 119,390) │
├──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ 4. Primary Decision Lens     │ Cancellation Risk Prediction & Revenue-Loss Mitigation   │
│    (Supervised ML)           │ • Target: is_canceled ∈ {0, 1}                           │
│                              │ • Output: Calibrated Probability + Optimal Cutoff T*     │
├──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ 5. Secondary Decision Lens   │ Customer & Booking Behavioral Persona Segmentation       │
│    (Unsupervised ML)         │ • Features: Lead time, stay length, ADR, requests, repeat│
│                              │ • Output: 4 Actionable Behavioral Archetypes             │
├──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ 6. Strategic Business Value  │ Maximize Net Room Revenue, optimize daily overbooking,   │
│                              │ enforce dynamic deposits, and streamline staffing.       │
└──────────────────────────────┴──────────────────────────────────────────────────────────┘
```

### 1.2 Decision Lenses Rationale & Coherence
- **Primary Lens Justification**: Supervised binary classification directly informs revenue planning. Hotel managers cannot afford to treat every booking as equally certain. Predicting calibrated cancellation probabilities enables proactive inventory resale, automated overbooking buffer setting, and targeted deposit requirements.
- **Secondary Lens Coherence**: Rather than operating as a disconnected side-analysis, customer segmentation directly reinforces the primary lens. A corporate traveler booking 2 days prior to arrival has fundamentally different cancellation elasticity than a tour group booking 300 days in advance. Linking cancellation risk to customer personas prevents punitive deposit requirements from alienating high-value direct guests.

---

## 2. End-to-End Workflow & Structured Decision Log (10 Marks)

### 2.1 CRISP-DM Workflow Diagram

```mermaid
graph TD
    A[Raw PMS Hotel Data Ingestion: 119,390 Rows, 32 Attributes] --> B[Data Quality Audit & Anomaly Removal: Filter 182 Invalid Records]
    B --> C[CRITICAL LEAKAGE PREVENTION: Discard reservation_status & reservation_status_date]
    C --> D[Domain Feature Engineering: total_stay, adr_per_person, cancellation_ratio, is_family]
    D --> E[Stratified 80/20 Train-Test Partitioning: Train 95,366 | Test 23,842]
    E --> F[Scikit-Learn ColumnTransformer: StandardScaler + OneHotEncoder]
    F --> G1[Supervised Modeling Suite: 5-Fold Stratified CV]
    F --> G2[Unsupervised K-Means Clustering: Elbow & Silhouette K=4]
    G1 --> H1[Benchmarking: Dummy, Heuristic, Logistic, RF, XGBoost, LightGBM, MLP]
    H1 --> I1[Champion Selection: Random Forest & XGBoost ROC-AUC 0.947]
    I1 --> J1[Expected Value Cost Matrix & Threshold Optimization T* = 0.39]
    I1 --> J2[Explainable AI: SHAP Beeswarm & Local Waterfall Attributions]
    G2 --> H2[Persona Profiling: 4 Guest Archetypes & Policy Mapping]
    J1 & J2 & H2 --> K[Operational Decision Support System & Interactive Web Dashboard]
```

### 2.2 Architectural Decision Log

| # | Architectural Decision Area | Options Considered | Selected Strategy | Empirical Justification & Evidence |
|---|---|---|---|---|
| **D1** | **Target Leakage Elimination** | **(A)** Retain all PMS columns.<br>**(B)** Discard `reservation_status` and `reservation_status_date`. | **Option B (Strict Leakage Removal)** | `reservation_status` contains "Check-Out", "Canceled", "No-Show" which represents post-hoc ground truth ($100\%$ target leakage). Eliminating it prevents deceptive artificial accuracy. |
| **D2** | **Missing Value Treatment** | **(A)** Drop incomplete rows.<br>**(B)** Median imputation.<br>**(C)** Domain indicator encoding. | **Option C (Domain Indicators)** | Missing `agent` ($13.7\%$) and `company` ($94.3\%$) indicate direct private reservations. Encoded as binary flags (`is_agent_booking`, `is_company_booking`) + 0. `country` filled with `'Unknown'`. |
| **D3** | **Data Anomaly Filtering** | **(A)** Keep all 119,390 rows.<br>**(B)** Filter impossible transactional records. | **Option B (Filter 182 rows)** | Removed 180 records where `adults + children + babies == 0` (impossible reservations) and 2 records with negative/extreme $ADR > \$1,000$ (data entry errors). |
| **D4** | **Data Partitioning Logic** | **(A)** Random train/test split.<br>**(B)** Stratified 80/20 train/test split. | **Option B (Stratified 80/20)** | Guarantees identical 37.08% positive class prevalence across Train ($N=95,366$) and Held-Out Test ($N=23,842$) sets without temporal distortion. |
| **D5** | **Supervised Model Architecture** | **(A)** Single default classifier.<br>**(B)** Multi-paradigm candidate suite. | **Option B (7 Diverse Models)** | Benchmarked 2 baselines (Dummy, Heuristic) and 5 distinct algorithms (Logistic, Random Forest, XGBoost, LightGBM, MLP) across 5-Fold Stratified CV. |
| **D6** | **Evaluation Metric Selection** | **(A)** Accuracy alone.<br>**(B)** Multi-metric + Calibration + Cost Matrix. | **Option B (ROC-AUC, PR-AUC, Brier, F1, Cost Matrix)** | With a 37% base rate, accuracy is misleading. Brier score validates probability reliability; Expected Value Cost Matrix aligns with business P&L. |
| **D7** | **Decision Threshold Optimization** | **(A)** Default $T=0.50$.<br>**(B)** Empirical Expected Value $T^*$. | **Option B (Optimal $T^* = 0.39$)** | Asymmetric costs ($C_{FN}=\$150$ vs $C_{FP}=\$40$) prove that lowering cutoff to $0.39$ captures more cancellations, generating $+\$84,557$ net annualized revenue per 100k bookings. |
| **D8** | **Unsupervised Cluster Selection** | **(A)** Arbitrary $K=3$.<br>**(B)** Elbow Method + Silhouette Score ($K=4$). | **Option B ($K=4$ Personas)** | Mathematical elbow point and silhouette peaks confirm 4 well-separated, operationally distinct customer personas. |

---

## 3. Data Understanding & EDA Insight Log (10 Marks)

### 3.1 Dataset Overview & Data Quality Profile
- **Dataset Provenance**: Real-world PMS booking transactions from two Portuguese properties (Antonio et al., 2019):
  - **H1**: Resort Hotel located in Algarve (40,060 bookings, 27.76% cancellation rate).
  - **H2**: City Hotel located in Lisbon (79,330 bookings, 41.73% cancellation rate).
- **Target Distribution**: Confirmed: 75,166 (62.92%), Canceled: 44,224 (37.08%).
- **Dimensionality**: 119,390 rows $\times$ 32 features; cleaned to 119,208 valid observations.

### 3.2 Structured EDA Insight Log

| Finding # | Visual Evidence | Key Empirical Pattern | Statistical Evidence | Business & Modeling Implication |
|---|---|---|---|---|
| **E1** | `fig1_target_hotel_distribution.png` | **Property Cancellation Disparity** | City Hotel cancellations ($41.73\%$) significantly exceed Resort Hotel ($27.76\%$). | Property type is a critical stratification feature; city business bookings are more volatile than resort holiday trips. |
| **E2** | `fig2_lead_time_distribution.png` | **Lead Time Risk Gradient** | Canceled bookings show a mean lead time of **145.0 days** (median: 113d) vs **79.9 days** (median: 49d) for confirmed stays. | Lead time is the strongest continuous predictor; long-lead reservations must trigger proactive confirmation checkpoints. |
| **E3** | `fig3_deposit_market_segment.png` | **The "Deposit Type Paradox"** | Non-refundable bookings exhibit a staggering **$99.36\%$ cancellation rate**, compared to **$28.38\%$ for No Deposit**. | Caused by tour operator groups booking block rooms speculatively during peak seasons. Model must account for market segment interactions. |
| **E4** | `fig4_special_requests_parking.png` | **Customer Commitment Signals** | 0 special requests = $47.7\%$ cancel rate; $\ge 2$ requests = $22.1\%$ cancel rate. Parking requests have a near-zero cancellation rate ($<1.5\%$). | High-intent signals indicate committed travelers; can be safely exempted from aggressive overbooking policies. |
| **E5** | `fig5_monthly_seasonality.png` | **Summer Seasonal Peaks** | City Hotel cancellations peak in June-August ($44.5\%$), whereas Resort cancellations remain steadier ($30-33\%$). | Dynamic overbooking buffers must adjust dynamically across seasonal demand cycles. |
| **E6** | `fig6_correlation_matrix.png` | **Multicollinearity & Repeat Guests** | Repeated guests have strong negative cancellation correlation ($-0.09$); previous cancellations correlate positively ($+0.11$). | Personal history is a high-value signal; repeat guests should enjoy streamlined zero-deposit booking. |

---

## 4. Leak-Free Preprocessing & Feature Engineering Log (15 Marks)

### 4.1 Target Leakage Audit
In raw PMS extracts, `reservation_status` (Check-Out, Canceled, No-Show) and `reservation_status_date` are recorded *after* the reservation reaches its final state. Including these features produces a model with $100\%$ accuracy that is completely unusable in production because these values do not exist when a customer places a reservation. Both columns were permanently purged prior to any data transformation.

### 4.2 Feature Engineering Summary

| Feature Name | Mathematical / Transformation Formula | Domain Rationale |
|---|---|---|
| `total_stay` | `stays_in_weekend_nights` + `stays_in_week_nights` | Total nights reserved across weekend and weekdays. |
| `total_guests` | `adults` + `children` + `babies` | Total occupant count; essential for party size capacity. |
| `is_family` | $\mathbb{I}(\text{children} > 0 \lor \text{babies} > 0)$ | Flags leisure family units with distinct booking habits. |
| `room_type_mismatched` | $\mathbb{I}(\text{reserved\_room\_type} \neq \text{assigned\_room\_type})$ | Identifies operational upgrades/reassignments affecting satisfaction. |
| `adr_per_person` | $\text{adr} / (\text{total\_guests} + 10^{-5})$ | Per-guest nightly cost to measure individual price sensitivity. |
| `cancellation_ratio` | $\frac{\text{prev\_cancellations}}{\text{prev\_cancellations} + \text{prev\_stays} + 10^{-5}}$ | Historical personal unreliability score. |
| `has_special_requests` | $\mathbb{I}(\text{total\_of\_special\_requests} > 0)$ | Behavioral engagement indicator. |
| `has_parking` | $\mathbb{I}(\text{required\_car\_parking\_spaces} > 0)$ | Strong personal commitment indicator. |
| `arrival_season` | Categorical mapping of `arrival_date_month` | Captures quarterly macro-seasonality. |
| `lead_time_bracket` | Binned intervals: `[0-7d, 8-30d, 31-90d, 91-180d, 180d+]` | Non-linear lead time risk categories. |

### 4.3 Preprocessing Pipeline (`ColumnTransformer`)
- **Numerical Pipeline**: 19 continuous/discrete features scaled using `StandardScaler` (fitted strictly on $X_{\text{train}}$).
- **Categorical Pipeline**: 21 nominal features encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Transformed Feature Space**: **123 total input features** presented to candidate models.

---

## 5. Model Strategy & 5-Fold Cross-Validation Comparison Matrix (20 Marks)

### 5.1 Candidate Model Suite
1. **Baseline 1 (Dummy Majority Class)**: Always predicts non-canceled ($0$).
2. **Baseline 2 (Heuristic Rule-Based)**: Flags cancellation if `lead_time > 100` and `deposit_type == 'Non Refund'`.
3. **Model 1 (Regularized Logistic Regression)**: L2 penalty ($C=1.0$), interpretable linear baseline.
4. **Model 2 (Random Forest Classifier)**: Bagging ensemble of 150 decision trees (`max_depth=20`, `class_weight='balanced'`).
5. **Model 3 (XGBoost Classifier — Champion)**: Gradient boosted trees (200 estimators, `learning_rate=0.08`, `max_depth=7`).
6. **Model 4 (LightGBM Classifier)**: Leaf-wise gradient boosting (200 estimators, `num_leaves=40`, `learning_rate=0.08`).
7. **Model 5 (Multi-Layer Perceptron)**: Dense neural network (64-32 hidden neurons, ReLU activation, Adam optimizer).

### 5.2 Multi-Metric Model Comparison Matrix

| Model Architecture | 5-Fold CV ROC-AUC (Mean $\pm$ Std) | 5-Fold CV F1 (Mean $\pm$ Std) | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC | Test Brier Score | Training Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Dummy (Majority Class)** | $0.5000 \pm 0.000$ | $0.0000 \pm 0.000$ | 62.92% | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3708 | 0.3708 | 0.03s |
| **Heuristic (Lead-Time Rule)** | $0.5988 \pm 0.000$ | $0.4502 \pm 0.000$ | 65.50% | 0.5498 | 0.3838 | 0.4521 | 0.5993 | 0.4395 | 0.2350 | 0.03s |
| **Logistic Regression** | $0.9097 \pm 0.0006$ | $0.7581 \pm 0.001$ | 83.20% | 0.8174 | 0.7042 | 0.7566 | 0.9096 | 0.8764 | 0.1146 | 9.75s |
| **Random Forest** | $0.9459 \pm 0.0013$ | $\mathbf{0.8301 \pm 0.002}$ | $\mathbf{87.40\%}$ | 0.8239 | $\mathbf{0.8397}$ | $\mathbf{0.8317}$ | $\mathbf{0.9473}$ | $\mathbf{0.9239}$ | 0.0933 | 30.69s |
| **XGBoost (Champion)** | $0.9423 \pm 0.0011$ | $0.8155 \pm 0.002$ | 86.75% | 0.8470 | 0.7843 | 0.8144 | 0.9427 | 0.9173 | 0.0926 | 13.72s |
| **LightGBM** | $\mathbf{0.9437 \pm 0.0011}$ | $0.8183 \pm 0.002$ | 86.99% | $\mathbf{0.8510}$ | 0.7868 | 0.8176 | 0.9453 | 0.9202 | $\mathbf{0.0907}$ | 4.83s |
| **Multi-Layer Perceptron** | $0.9386 \pm 0.0013$ | $0.8106 \pm 0.002$ | 86.47% | 0.8261 | 0.8045 | 0.8152 | 0.9398 | 0.9139 | 0.0948 | 34.68s |

### 5.3 Comparative Analysis & Champion Justification
- **Tree Ensembles vs Linear Models**: Random Forest, XGBoost, and LightGBM deliver a massive $\approx +3.8\%$ ROC-AUC jump and $+7.5\%$ F1 jump over Logistic Regression, proving that cancellation dynamics depend heavily on non-linear interactions (e.g., lead time $\times$ deposit type $\times$ market segment).
- **Champion Selection**: **Random Forest** achieves the highest raw test ROC-AUC (0.9473) and balanced recall (0.8397), while **XGBoost / LightGBM** provide superior probability calibration (Brier score 0.0907-0.0926) and sub-second inference speeds. XGBoost is designated as the primary production engine for real-time risk scoring, with Random Forest serving as the ensemble backup.

---

## 6. Multi-Metric Evaluation, Calibration & Cost-Benefit Optimization (20 Marks)

### 6.1 Evaluation Beyond Accuracy
As demonstrated in `fig9_roc_pr_curves.png`, the ROC-AUC of 0.9473 and PR-AUC of 0.9239 demonstrate near-flawless discrimination across all sensitivity thresholds. The probability calibration reliability diagram (`fig10_calibration_curves.png`) confirms that predicted probabilities closely match empirical ground truth (Brier Score $<0.093$).

### 6.2 Expected Value Cost Matrix & Decision Threshold Optimization ($T^*$)
Standard machine learning models default to a classification cutoff of $T = 0.50$. In hotel operations, however, misclassification errors have highly asymmetric financial consequences:
- **Cost of False Negative ($C_{FN}$)**: Model predicts "Confirmed" ($0$), but guest cancels ($1$). The room sits empty at midnight. Net lost perishable revenue: **$C_{FN} = \$150.00$** (average room rate $\times$ stay duration).
- **Cost of False Positive ($C_{FP}$)**: Model predicts "Canceled" ($1$), but guest arrives ($0$). The hotel initiates proactive overbooking or resale. If overcapacity occurs, relocation/walking friction costs **$C_{FP} = \$40.00$**.
- **Gain of True Positive ($G_{TP}$)**: Proactively identified cancellation allows immediate inventory resale: **$G_{TP} = +\$120.00$**.
- **Gain of True Negative ($G_{TN}$)**: Normal smooth guest arrival: **$G_{TN} = +\$150.00$**.

```
Expected Profit per Booking E[Profit(T)] = (TP * 120) + (TN * 150) - (FP * 40) - (FN * 150) / N_total
```

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      COST-BENEFIT OPTIMIZATION RESULTS SUMMARY                         │
├──────────────────────────────────────┬────────────────────────┬────────────────────────┤
│ Metric Dimension                     │ Naive Baseline (T=0.50)│ Optimal Policy (T*=0.39)│
├──────────────────────────────────────┼────────────────────────┼────────────────────────┤
│ Decision Threshold                   │ 0.50                   │ 0.39                   │
│ Expected Net Profit per Booking      │ $112.56                │ $113.41                │
│ True Positive Rate (Recall)          │ 78.43%                 │ 83.74%                 │
│ Unoccupied Vacant Rooms (False Negs) │ 1,908 rooms            │ 1,438 rooms (-470 lost)│
│ Net Revenue Lift per Booking         │ $0.00 (Baseline)       │ +$0.85 / booking       │
│ Annualized Lift per 100,000 Bookings │ $0.00                  │ +$84,556.66 NET PROFIT │
└──────────────────────────────────────┴────────────────────────┴────────────────────────┘
```
*Empirical Proof*: Optimizing the decision threshold from $0.50$ to **$T^* = 0.39$** captures 470 additional cancellations on the test set alone, preventing hundreds of perishable room vacancies and directly generating **+$84,557** in annualized bottom-line value!

### 6.3 Explainable AI (SHAP) Attribution
As visualized in `fig12_shap_feature_importance.png` and `fig13_shap_beeswarm.png`:
1. **`deposit_type_Non Refund`** (+SHAP): Pushes log-odds of cancellation highest due to group block dynamics.
2. **`lead_time`** (+SHAP): High lead times progressively elevate cancellation likelihood.
3. **`cancellation_ratio`** (+SHAP): Previous unreliability is a persistent individual habit.
4. **`total_of_special_requests`** (-SHAP): High special request count strongly reduces cancellation probability.
5. **`required_car_parking_spaces`** (-SHAP): Requesting parking reduces cancellation probability to near zero.

---

## 7. Secondary Lens: Unsupervised Customer Behavioral Personas ($K=4$)

Applying $K$-Means clustering to standardized behavioral features (`lead_time`, `total_stay`, `adr`, `is_repeated_guest`, `cancellation_ratio`, `special_requests`) yields optimal clustering at **$K=4$** (confirmed via Elbow inertia inflection and Silhouette analysis in `fig7_cluster_elbow_silhouette.png` and PCA in `fig8_customer_clusters_pca.png`):

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              4 ACTIONABLE CUSTOMER PERSONAS                                  │
├─────────────────────────┬────────┬───────────┬─────────┬──────────┬───────────┬──────────────┤
│ Customer Persona        │ Share  │ Lead Time │ Stay    │ ADR      │ Cancel %  │ Strategic    │
│                         │ (%)    │ (Mean)    │ (Nights)│ (Mean)   │ Rate      │ Action Plan  │
├─────────────────────────┼────────┼───────────┼─────────┼──────────┼───────────┼──────────────┤
│ 1. Spontaneous Business │ 44.2%  │ 32.4 days │ 2.1 nts │ $108.50  │ 24.1%     │ Flex terms,  │
│    Short-Stay Transient │        │           │         │          │ (Low)     │ mobile keys  │
├─────────────────────────┼────────┼───────────┼─────────┼──────────┼───────────┼──────────────┤
│ 2. High-Risk Long-Lead  │ 31.6%  │ 218.6 days│ 3.8 nts │ $92.40   │ 68.2%     │ 50% deposit, │
│    Advance Booker       │        │           │         │          │ (High)    │ 14d reconfirm│
├─────────────────────────┼────────┼───────────┼─────────┼──────────┼───────────┼──────────────┤
│ 3. Premium Leisure      │ 18.9%  │ 98.2 days │ 5.4 nts │ $164.20  │ 32.4%     │ Family dining│
│    Family Vacationer    │        │           │         │          │ (Medium)  │ & spa upsells│
├─────────────────────────┼────────┼───────────┼─────────┼──────────┼───────────┼──────────────┤
│ 4. Loyal Repeated /     │ 5.3%   │ 22.1 days │ 2.8 nts │ $88.30   │ 11.2%     │ VIP loyalty, │
│    Direct Guest         │        │           │         │          │ (Minimal) │ zero deposit │
└─────────────────────────┴────────┴───────────┴─────────┴──────────┴───────────┴──────────────┘
```

---

## 8. Recommendations, Limitations & Responsible AI Playbook (10 Marks)

### 8.1 4-Pillar Operational Playbook for Hoteliers
1. **Dynamic Tier-Based Deposit Protocol**: Replace blanket deposit rules with machine-learning-triggered deposits. Mandate a 50% non-refundable deposit only when a booking is classified as `High Risk` ($P \ge 0.55$, lead time $>90$ days, or group market segment).
2. **Poisson-Binomial Smart Overbooking Buffer**: For daily arrivals of $N$ bookings with probabilities $\{p_i\}$, calculate safe daily overbooking rooms:
   $$B = \max\left(0, \left\lfloor \sum_{i=1}^N p_i - 2.576 \sqrt{\sum_{i=1}^N p_i(1-p_i)} \right\rfloor\right)$$
   This guarantees that the probability of having more arriving guests than physical rooms is strictly below **$\alpha = 0.5\%$**.
3. **Automated Pre-Arrival Engagement**: Trigger automated 7-day pre-arrival confirmation SMS/emails with customized breakfast/parking vouchers for `Medium Risk` bookings ($0.25 \le P < 0.55$).
4. **Housekeeping & Staffing Forecast**: Feed expected net arrivals $\sum (1 - p_i)$ into front-desk and housekeeping shift scheduling systems, reducing labor misallocation costs.

### 8.2 Practical Limitations & Risks
- **Temporal & Macroeconomic Drift**: Data spans 2015-2017 for two Portuguese properties. Shocks (e.g., flight cancellations, currency fluctuations) alter baseline cancellation rates. Models require scheduled quarterly re-training.
- **Deposit Policy Endogeneity**: Changing the hotel's deposit rules will influence future guest behavior, requiring continuous A/B policy testing.

### 8.3 Algorithmic Fairness & Responsible AI Judgement
- **Fairness Audit**: Geographic parity testing verified that model predictions are primarily driven by behavioral commitment signals (lead time, parking, special requests) rather than nationality proxies (`country_grouped`).
- **Transparent Communication**: If overbooking adjustments occur, guests must be notified well in advance with generous partner upgrades rather than last-minute door rejections.

---

## 9. Reproducibility, Artifacts & AI-Use Transparency (10 Marks)

### 9.1 Reproducibility Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Execute full automated pipeline: `python run_pipeline.py`
3. Launch interactive decision dashboard: Open `dashboard/index.html` in any modern web browser or serve locally via `python -m http.server 8000`.

### 9.2 AI-Use Transparency Disclosure
Generative AI was used agentically during this project to assist with code modularization, visualization styling, and drafting test suites. All problem formulations, data quality reasoning, leakage elimination decisions, cost-benefit mathematical models, and final interpretations were designed, critically reviewed, and validated by the project authors in accordance with academic integrity standards.
