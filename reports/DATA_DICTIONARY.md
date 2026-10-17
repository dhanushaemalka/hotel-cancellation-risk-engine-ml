# Hotel Booking Demand — Comprehensive Data Dictionary
**Domain 6: Tourism & Hospitality**  
**Course**: IT3091 Machine Learning  
**Dataset Reference**: Antonio, N., de Almeida, A. and Nunes, L. (2019) *"Hotel booking demand datasets"*, *Data in Brief*, 22, pp. 41–49.

---

## 1. Overview & Dataset Provenance
- **Source**: Property Management Systems (PMS) of two actual hotel properties located in Portugal:
  - **H1**: A 4-star Resort Hotel located in the Algarve region (40,060 observations).
  - **H2**: A 4-star City Hotel located in the Lisbon metropolitan area (79,330 observations).
- **Observation Period**: Bookings scheduled to arrive between **July 1, 2015** and **August 31, 2017**.
- **Raw Volume**: 119,390 rows $\times$ 32 features.
- **Target Variable**: `is_canceled` (Binary: 1 = Canceled/No-Show, 0 = Stayed / Checked-Out).

---

## 2. Complete Attribute Definitions & Specifications

| # | Attribute Name | Data Type | Value Domain / Example Values | Business Definition & Operational Significance | Handling & Modeling Status |
|---|---|---|---|---|---|
| 1 | `hotel` | Categorical | `Resort Hotel`, `City Hotel` | Property type. Resort and city properties have divergent demand elasticity, cancellation rates, and seasonality. | Feature (One-Hot Encoded) |
| 2 | `is_canceled` | Binary (Target) | `0` (Not Canceled), `1` (Canceled) | **Ground Truth Target Variable**: Denotes whether reservation was cancelled prior to check-in or resulted in a no-show. Baseline positive class rate $\approx 37.0\%$. | **Supervised Target ($y$)** |
| 3 | `lead_time` | Numerical (Integer) | $0 \dots 737$ days | Number of days between PMS booking entry date and guest arrival date. Strongest single predictor of cancellation risk. | Scaled & Bucketed Feature |
| 4 | `arrival_date_year` | Numerical (Discrete) | `2015`, `2016`, `2017` | Year of scheduled arrival date. | Feature / Temporal Tracking |
| 5 | `arrival_date_month` | Categorical | `January` $\dots$ `December` | Month of scheduled arrival. Crucial for capturing annual leisure and business seasonality cycles. | Feature (Mapped to Season & Encoded) |
| 6 | `arrival_date_week_number` | Numerical (Discrete) | $1 \dots 53$ | ISO week number of the arrival date. Granular seasonality indicator. | Numerical Scaled Feature |
| 7 | `arrival_date_day_of_month` | Numerical (Discrete) | $1 \dots 31$ | Day of arrival month. Useful for intra-month payroll/travel timing effects. | Numerical Scaled Feature |
| 8 | `stays_in_weekend_nights` | Numerical (Integer) | $0 \dots 19$ | Count of weekend nights (Saturday and/or Sunday) reserved. | Engineered into `total_stay` |
| 9 | `stays_in_week_nights` | Numerical (Integer) | $0 \dots 50$ | Count of week nights (Monday through Friday) reserved. | Engineered into `total_stay` |
| 10 | `adults` | Numerical (Integer) | $0 \dots 55$ | Number of adult occupants. Records where adults + children + babies = 0 are filtered as invalid. | Feature & Guest Sum |
| 11 | `children` | Numerical (Float) | $0.0 \dots 10.0$ | Number of children. Contains 4 missing values in raw data. | Imputed (0) & Feature |
| 12 | `babies` | Numerical (Integer) | $0 \dots 10$ | Number of infant occupants. | Feature & Family Flag |
| 13 | `meal` | Categorical | `BB` (Bed & Breakfast), `HB` (Half Board), `FB` (Full Board), `SC`/`Undefined` (Room Only) | Meal plan booked. `Undefined` mapped to `SC` (Self Catering / No Meal). | Feature (One-Hot Encoded) |
| 14 | `country` | Categorical (ISO) | `PRT`, `GBR`, `FRA`, `ESP`, `DEU`, $\dots$ (177 codes) | Country of origin of primary guest. Missing values ($0.4\%$) filled with `'Unknown'`. Low-frequency countries bucketed to `'Other'`. | Feature (Top 20 + Other Encoded) |
| 15 | `market_segment` | Categorical | `Direct`, `Corporate`, `Online TA`, `Offline TA/TO`, `Groups`, `Complementary`, `Aviation` | Distribution market segment. Key differentiator for cancellation risk (Groups show distinct behavior). | Feature (One-Hot Encoded) |
| 16 | `distribution_channel` | Categorical | `Direct`, `Corporate`, `TA/TO`, `GDS`, `Undefined` | Intermediary booking vehicle. `Undefined` mapped to `TA/TO`. | Feature (One-Hot Encoded) |
| 17 | `is_repeated_guest` | Binary (Integer) | `0` (First-time), `1` (Repeat) | Indicates if guest has stayed previously. Strong loyalty signal with low cancellation rates ($<15\%$). | Feature (Binary Indicator) |
| 18 | `previous_cancellations` | Numerical (Integer) | $0 \dots 26$ | Number of previous bookings cancelled by the customer prior to current reservation. | Engineered into `cancellation_ratio` |
| 19 | `previous_bookings_not_canceled` | Numerical (Integer) | $0 \dots 72$ | Number of previous successful stays completed by the customer. | Engineered into `cancellation_ratio` |
| 20 | `reserved_room_type` | Categorical (Code) | `A` $\dots$ `L`, `P` | Room category code requested at reservation time. | Feature (Categorical) |
| 21 | `assigned_room_type` | Categorical (Code) | `A` $\dots$ `L`, `P` | Room category code allocated by front desk at check-in or room blocking. | Engineered into `room_mismatch` |
| 22 | `booking_changes` | Numerical (Integer) | $0 \dots 21$ | Number of amendments made to reservation (dates, room type, guest count) prior to arrival. | Feature (Scaled) |
| 23 | `deposit_type` | Categorical | `No Deposit`, `Non Refund`, `Refundable` | Deposit guarantee terms. Exhibits the "Deposit Paradox" where `Non Refund` is heavily used by high-risk tour groups. | Feature (One-Hot Encoded) |
| 24 | `agent` | Categorical / ID | $1 \dots 535$, `NULL` (13.7%) | ID of travel agency making reservation. Missingness indicates direct reservation. | Binary Indicator `is_agent_booking` |
| 25 | `company` | Categorical / ID | $6 \dots 543$, `NULL` (94.3%) | ID of corporate entity booking/paying. Missingness indicates non-corporate booking. | Binary Indicator `is_company_booking` |
| 26 | `days_in_waiting_list` | Numerical (Integer) | $0 \dots 391$ | Number of days booking was held on waiting list prior to confirmation. | Feature (Scaled) |
| 27 | `customer_type` | Categorical | `Transient`, `Transient-Party`, `Contract`, `Group` | Booking behavioral categorization. | Feature (One-Hot Encoded) |
| 28 | `adr` | Numerical (Continuous) | $-\$6.38 \dots \$5,400.00$ | **Average Daily Rate**: Lodging revenue divided by total paid room nights. Filtered for anomalies ($ADR < 0$ or $ADR > \$1,000$). | Feature (Scaled & `adr_per_person`) |
| 29 | `required_car_parking_spaces` | Numerical (Integer) | $0 \dots 8$ | Number of vehicle parking spots requested. Powerful commitment signal (near-zero cancellations when $>0$). | Feature (Scaled & Binary Flag) |
| 30 | `total_of_special_requests` | Numerical (Integer) | $0 \dots 5$ | Number of special preferences requested (high floor, twin beds, early check-in). Strong commitment indicator. | Feature (Scaled & Binary Flag) |
| 31 | `reservation_status` | Categorical (LEAKAGE) | `Check-Out`, `Canceled`, `No-Show` | **CRITICAL TARGET LEAKAGE**: Post-event recorded status. | **STRICTLY EXCLUDED / DROPPED** |
| 32 | `reservation_status_date` | Date (LEAKAGE) | `2014-10-17` $\dots$ `2017-09-14` | **CRITICAL TARGET LEAKAGE**: Date when final status was set. Directly reveals cancellation timing. | **STRICTLY EXCLUDED / DROPPED** |

---

## 3. Engineered Features Summary

| Engineered Feature | Mathematical / Logical Formula | Business Rationality |
|---|---|---|
| `total_stay` | `stays_in_weekend_nights` + `stays_in_week_nights` | Captures overall trip length of stay. |
| `total_guests` | `adults` + `children` + `babies` | Overall party size; used for occupancy and staffing. |
| `is_family` | $\mathbb{I}(\text{children} > 0 \lor \text{babies} > 0)$ | Flags leisure family vacationers with distinct lead times and meal needs. |
| `room_type_mismatched` | $\mathbb{I}(\text{reserved\_room\_type} \neq \text{assigned\_room\_type})$ | Operational upgrades or downgrades affecting guest satisfaction. |
| `adr_per_person` | $\text{adr} / (\text{total\_guests} + 10^{-5})$ | Normalizes room rate by guest count to measure individual price sensitivity. |
| `cancellation_ratio` | $\frac{\text{prev\_cancellations}}{\text{prev\_cancellations} + \text{prev\_stays} + 10^{-5}}$ | Historical personal unreliability score. |
| `has_special_requests` | $\mathbb{I}(\text{total\_of\_special\_requests} > 0)$ | Behavioral engagement flag. |
| `has_parking` | $\mathbb{I}(\text{required\_car\_parking\_spaces} > 0)$ | High-intent commitment signal. |
| `arrival_season` | $\text{Month} \to \{\text{Winter}, \text{Spring}, \text{Summer}, \text{Autumn}\}$ | Aggregates seasonal demand swings. |
| `lead_time_bracket` | Quantile bins: `[0-7d, 8-30d, 31-90d, 91-180d, 180d+]` | Non-linear lead time risk zones for revenue managers. |
