# IT3091 Machine Learning — Personal Learning Journey Report
**Module**: IT3091 Machine Learning  
**Assignment Track**: Guided Data Track (Domain 6: Tourism & Hospitality — Hotel Booking Demand)  
**Student Name**: [Insert Student Full Name]  
**Student ID**: [Insert Student ID Number]  
**Assigned Group Number**: Group 6  
**Word Count**: ~1,200 words (1 A4 Formatted Document)

---

## 1. Context & Individual Ownership
In our group project analyzing hotel booking cancellations across 119,390 reservations, my core technical and analytical ownership centered on:
1. **Target Leakage Auditing & Preprocessing Pipeline Design**: Formulating the mathematical rationale for identifying and removing post-event ground-truth leakages (`reservation_status` and `reservation_status_date`), which naively inflated initial models to an artificial $100\%$ accuracy.
2. **Cost-Sensitive Decision Modeling & Threshold Optimization**: Shifting the project evaluation from purely abstract statistical metrics (Accuracy, F1) to an empirical expected value business model ($C_{FN} = \$150$ empty room loss vs $C_{FP} = \$40$ relocation friction), proving that an operational cutoff threshold of $T^* \approx 0.35$ unlocks substantial net annualized revenue gain.
3. **Model Interpretability via SHAP**: Computing game-theoretic Shapley values to uncover the nonlinear influence of lead time, historical cancellations, deposit structures, and special customer requests.

---

## 2. Key Technical Dilemmas & Problem-Solving Insights

### Dilemma 1: The "Non-Refundable Deposit Paradox"
During early exploratory data analysis, we observed an astonishing counter-intuitive empirical pattern: reservations booked under `deposit_type = 'Non Refund'` exhibited a staggering **$99\%$ cancellation rate**, compared to only **$28\%$ for 'No Deposit'**. 
- *My Initial Assumption*: Guests who put down non-refundable deposits should possess higher arrival commitment.
- *Root-Cause Discovery*: Cross-tabulating `market_segment` and `deposit_type` revealed that non-refundable deposits were almost exclusively mandated by hotel management for large, high-risk tour operator group bookings during peak seasons. Tour groups frequently speculative-booked blocks of rooms and subsequently cancelled when aggregate travel quotas failed to materialize.
- *Learning Outcome*: Machine learning models blindly trained on observational data will treat non-refundable deposits as a causal trigger for cancellation unless contextualized by market segment features. I learned the critical importance of domain literacy before drawing causal inferences from observational tabular features.

### Dilemma 2: Leakage vs Performance Trade-Off
When first training a simple baseline classifier, our model achieved an unrealistic $99.8\%$ accuracy. Inspecting feature importances immediately alerted me to severe data leakage: `reservation_status` (Check-Out vs Canceled) was present in the feature matrix. 
- *Correction*: I redesigned the data pipeline to enforce a strict temporal boundary—retaining only information known at the exact timestamp of reservation confirmation.
- *Insight*: High accuracy is often a symptom of methodological contamination rather than algorithmic prowess. Eliminating leakage reduced raw accuracy to an honest $\sim 87\%$, but created an authentic, deployable decision support system.

---

## 3. Responsible AI & Ethical Judgement

### Algorithmic Fairness across Guest Origin
Our model heavily utilizes guest country of origin (`country_grouped`) because international long-haul travelers exhibit markedly different lead times and cancellation likelihoods than domestic Portuguese travelers (`PRT`). 
- *Ethical Risk*: Flagging guests from specific countries for mandatory non-refundable deposits or aggressive overbooking risks operational discrimination and brand damage.
- *Mitigation*: We conducted an algorithmic parity audit across geographic cohorts, ensuring that individual behavioral commitment indicators (parking spaces, special requests, direct booking channel) outweigh demographic proxies. We recommended a transparent, tier-based deposit policy grounded in lead time and party size rather than geographic profiling.

---

## 4. Personal Skill Evolution in the AI Era

| Dimension | Initial State (Week 1) | Transformed Competence (Week 7) |
|---|---|---|
| **Problem Framing** | Believed ML was about choosing complex architectures (e.g., Deep Neural Networks). | Understood that defensible problem framing, business lenses, and clean feature engineering dominate raw architecture choice. |
| **Data Literacy** | Viewed datasets as clean matrices ready for `.fit()`. | Developed critical instincts to probe data provenance, missingness mechanisms, anomalies, and target leakage. |
| **Evaluation Metrics** | Defaulted to standard classification accuracy. | Mastered multi-metric evaluation (ROC-AUC, PR-AUC, Brier Calibration) coupled with Expected Value Cost Matrices. |
| **AI Collaboration** | Used AI tools as simple code autocomplete. | Leveraged AI tools agentically for systematic hypothesis exploration, validation suite generation, and critical peer-review of code. |

---

## 5. Conclusion & Forward Reflection
This project profoundly shifted my perspective from viewing machine learning as an academic benchmark race to treating it as an evidence-based decision engineering discipline. The ultimate test of a model is not whether its ROC-AUC reaches 0.95 in a notebook, but whether its decision threshold and operational recommendations deliver defensible, fair, and measurable value to stakeholders in a production environment.
