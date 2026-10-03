# Data Dictionary — Commercial Auto Insurance Policy Dataset

| Column | Type | Description |
|--------|------|-------------|
| `policy_id` | string | Unique policy line identifier (POL-XXXXXX) |
| `insured_id` | string | Insured company identifier; one insured may have both APD and AL rows |
| `snapshot_date` | date | Date at which the policy record was captured; used for temporal analysis |
| `policy_effective_date` | date | Policy start date |
| `policy_expiration_date` | date | Policy end date (typically 1 year after effective date) |
| `coverage_type` | categorical | Coverage line: APD (Auto Physical Damage) or AL (Auto Liability) |
| `vehicle_count` | numeric | Number of vehicles in the insured's fleet |
| `vehicle_avg_age` | numeric | Average age of fleet vehicles in years |
| `driver_count` | numeric | Number of drivers listed on the policy |
| `driver_avg_age` | numeric | Average age of drivers in years |
| `years_in_business` | numeric | Number of years the insured company has operated |
| `prior_year_mileage_000` | numeric | Stated annual fleet mileage in thousands of miles, estimated at underwriting time |
| `business_type` | categorical | Type of business: Trucking, Retail, Construction, Service, Other |
| `state` | categorical | US state where the insured is headquartered |
| `prior_apd_claim_count` | numeric | Number of APD claims filed in the prior policy period |
| `prior_al_claim_count` | numeric | Number of AL claims filed in the prior policy period |
| `prior_loss_amount` | numeric | Total prior period loss amount in USD |
| `deductible` | numeric | Policy deductible in USD (250, 500, 1000, 2500, 5000) |
| `coverage_limit_000` | numeric | Policy coverage limit in thousands of USD |
| `annual_premium` | numeric | Annual premium charged in USD |
| `payment_frequency` | categorical | Premium payment frequency: Monthly, Quarterly, Annual |
| `risk_score_external` | numeric | External risk score from 0 (low risk) to 100 (high risk) |
| `has_safety_program` | boolean | 1 if the insured has a formal driver safety program, else 0 |
| `num_heavy_vehicles` | numeric | Count of heavy vehicles (trucks, semis) in the fleet |
| `late_payment_count` | numeric | Number of late premium payments in the current policy year |
| `first_claim_reported_date` | date | Date the first claim in the period was reported to the insurer; available after claim resolution, not at policy inception |
| `claim_paid_amount_current_period` | numeric | Amount paid by the insurer for claims in the current policy period; available after claim resolution, not at policy inception |
| `claim_status_current_period` | categorical | Status of claims in the current policy period (Open, Closed, Under Review, No Claim); available after claim resolution, not at policy inception |
| `days_to_first_claim_report` | numeric | Days between policy snapshot date and first claim report; available after claim resolution, not at policy inception |
| `claim_count` | numeric | Number of claims filed during the policy period; absent from score.csv |
| `total_loss_amount` | numeric | Total loss amount in USD during the policy period; 0.0 if no claims; absent from score.csv |