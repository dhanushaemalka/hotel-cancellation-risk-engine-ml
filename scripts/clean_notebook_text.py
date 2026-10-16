"""Replace damaged notebook prose with readable, accurate Markdown."""
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "notebooks" / "hotel_booking_demand_master.ipynb"
backup = path.with_name("hotel_booking_demand_master.before_text_cleanup.ipynb.bak")
if not backup.exists():
    shutil.copy2(path, backup)
executed = path.with_name("hotel_booking_demand_final_executed.ipynb")
executed_backup = executed.with_name("hotel_booking_demand_final_executed.before_text_cleanup.ipynb.bak")
if executed.exists() and not executed_backup.exists():
    shutil.copy2(executed, executed_backup)
nb = json.loads(path.read_text(encoding="utf-8-sig"))
markdown = {
0: '''# Hotel Booking Cancellation Prediction

IT3091 Machine Learning Project | Group 6 | Tourism and Hospitality

This project compares models for predicting booking cancellations and groups
guests by booking behaviour. The final examples show how model probabilities
can support deposit recommendations and daily overbooking estimates.

Dataset: Antonio, N., de Almeida, A. and Nunes, L. (2019),
“Hotel booking demand datasets”, Data in Brief, 22, pp. 41–49.

## Notebook contents
1. Project objective
2. Method choices
3. Data exploration
4. Data preparation
5. Model training
6. Test results
7. Decision threshold
8. Prediction explanations
9. Guest segmentation
10. Booking decision example
11. Recommendations and limitations
''',
2: '''## 1. Project objective

Hotels lose potential revenue when bookings are cancelled and rooms remain
empty. Accepting too many extra bookings can also leave a hotel without enough
rooms for arriving guests. This project explores how cancellation predictions
could help hotel managers balance these risks.

The dataset contains 119,390 bookings. About 37% were cancelled before cleaning.
Each row describes one booking, and the target is `is_canceled`:

- `1`: the booking was cancelled.
- `0`: the booking was not cancelled.

The main task is to estimate the probability that a booking will be cancelled.
The decision example uses this probability to assign a risk tier and suggest
an action. A separate clustering analysis groups bookings with similar guest
and stay characteristics.

The intended users are hotel revenue managers. Before using the model at a
specific stage of booking, they would need to confirm that every input feature
is available at that time.
''',
3: '''## 2. Method choices

| Area | Choice | Reason and interpretation |
|---|---|---|
| Outcome information | Remove `reservation_status` and `reservation_status_date` | These columns reveal the booking outcome. |
| Missing values | Fill missing children counts with 0 and countries with `Unknown`; create agent and company indicators | Preserve usable bookings and represent missing agent or company records explicitly. |
| Invalid records | Remove bookings with no guests and nightly prices outside 0–1,000 | The run removes 182 records. |
| Data split | Stratified 80/20 split, seed 42 | Use 95,366 bookings for training and 23,842 for testing, with similar target proportions. |
| Model comparison | Two baselines and five learned models | Compare simple rules with linear, bagging, boosting and neural models. The learned models use five-fold validation. |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC and Brier score | Examine classification performance and the quality of the probability estimates. |
| Operational model | Explicitly select XGBoost | This is a configured choice; Random Forest has the highest test accuracy in this run. |
| Decision threshold | Compare thresholds from 0.05 to 0.95 | Find the highest estimated value under the specified cost assumptions. The result is not evidence of realised hotel profit. |
| Guest groups | Fit four K-Means clusters and inspect elbow and silhouette plots | Summarise booking behaviour and display the groups using PCA. |

Preprocessing is fitted on the training split before cross-validation, and
the decision threshold is selected using test data. These limitations are
discussed at the end of the notebook.
''',
5: '''## 3. Data exploration

Load the booking data, check missing and invalid values, and inspect patterns
in cancellations, lead time, deposit type, requests, parking and seasonality.
The cells below generate the summary statistics and six charts.
''',
8: '''## 4. Data preparation

Clean invalid bookings and handle missing values. Create features for total
stay, guest count, family bookings, room mismatch, price per guest, past
cancellations, season and booking commitment.

Separate the target and make the stratified train/test split. Fit numerical
scaling and categorical encoding on the training data, then apply the fitted
transformations to the test data.
''',
10: '''## 5. Model training

Compare a majority-class baseline, a lead-time rule, Logistic Regression,
Random Forest, XGBoost, LightGBM and a neural network. The five learned models
use five-fold stratified cross-validation and are then fitted on the full
training split. The two baselines are evaluated directly on training data.

The fitted models are stored in `fitted_models` and saved under
`data/processed/models`.
''',
12: '''## 6. Test results

Evaluate every model on the held-out test split. The table includes accuracy,
precision, recall, F1, ROC-AUC, PR-AUC and Brier score. The plots compare ranking
performance and how closely predicted probabilities match observed outcomes.
''',
15: '''## 7. Decision threshold

Use the selected XGBoost model to compare estimated value across decision
thresholds. The example assumes a loss of 150 for a missed cancellation,
a loss of 40 for an incorrect cancellation flag, a gain of 120 for a correctly
flagged cancellation and a gain of 150 for a correctly predicted arrival.
These values are modelling assumptions, expressed in dollars.

The reported improvement compares the selected threshold with 0.50 under
those assumptions. It is a scenario estimate, not measured annual revenue.
''',
17: '''## 8. Prediction explanations

Calculate SHAP values for a sample of test bookings. The feature-importance
chart summarises the largest contributions to model predictions. The beeswarm
plot shows their direction and spread. These explanations describe the model;
they do not establish that a feature causes cancellations.
''',
19: '''## 9. Guest segmentation

Group bookings using lead time, stay duration, guest count, nightly price,
past cancellation ratio, requests, parking and repeated-guest status.
Inspect clustering scores, fit four groups, and summarise each group's booking
characteristics. PCA is used to display the groups in two dimensions.
''',
21: '''## 10. Booking decision example

Score one example reservation using the saved preprocessor and XGBoost model.
Display its cancellation probability, risk tier, deposit recommendation and
suggested action. Then estimate a conservative overbooking buffer for a sample
day with 200 bookings.

Risk tiers use fixed boundaries of 25% and 55%. The separate operational
action trigger uses the threshold calculated in Section 7.
''',
23: '''## 11. Recommendations and limitations

### Possible uses
- Review high-risk bookings for reconfirmation or a deposit requirement.
- Send pre-arrival reminders for medium-risk bookings.
- Use aggregated cancellation probabilities to support occupancy planning.
- Review guest groups when considering offers and service arrangements.

These actions are suggestions. Their effect on cancellations and customer
experience has not been tested in this project.

### Limitations
- The data covers two hotels in Portugal from 2015–2017. Results may differ
  for other hotels and for future bookings.
- The split is random rather than chronological. A later-period test would
  better assess predictions for future bookings.
- Country grouping is calculated before the split. Scaling and encoding are
  fitted before cross-validation rather than separately inside each fold.
- The decision threshold is tuned on the test set. A separate validation set
  should be used before reporting final business performance.
- Assigned room type and booking changes may be unavailable when a reservation
  is first created. Input availability needs to match the prediction time.
- Country grouping is recalculated when scoring an individual booking; the
  training mapping should instead be reused for consistent deployment.
- Deposit type and other inputs may reflect existing hotel policies. Their
  association with cancellation does not establish a causal effect.
- The overbooking buffer uses an independence assumption and a normal
  approximation. The stated risk tolerance is not a guaranteed guest walk rate.
- The original web dashboard uses a scoring formula rather than calling the
  trained model. The Python example above does use the saved model.
- No completed demographic fairness audit is demonstrated by this notebook.

### AI assistance
Generative AI assisted with brainstorming, code organisation and checking
parts of the project. This disclosure should be reviewed by the project team
to ensure it accurately describes their use and meets the course requirements.
'''
}
for index, text in markdown.items():
    assert nb["cells"][index]["cell_type"] == "markdown"
    nb["cells"][index]["source"] = text.splitlines(keepends=True)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        source = source.replace("# Step 9: Production Decision Support Smoke Test", "# Booking risk and daily overbooking example")
        source = source.replace("Optimal Cutoff Profit per Booking (T*=0.39)", "Optimal Cutoff Profit per Booking")
        cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
for cell in nb["cells"]:
    assert not any(ord(c) < 32 and c not in "\n\r\t" for c in "".join(cell["source"]))
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Cleaned notebook text and preserved backups.")
