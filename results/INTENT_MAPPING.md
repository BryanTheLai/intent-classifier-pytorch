# CLINC150 Intent Mapping

This document maps intent IDs (0-149) to their human-readable names in the CLINC150 dataset.

## Overview
- **Total Intents**: 150
- **Domains**: 10 (Banking, Credit Cards, Kitchen & Dining, Home, Auto & Commute, Travel, Utility, Work, Small Talk, Meta)
- **Model Accuracy**: 95.4%
- **Macro F1**: 0.954

## Intent List by ID

### Banking (IDs 0-12)
| ID | Intent Name |
|----|-------------|
| 0 | activate_my_card |
| 1 | age_limit |
| 2 | apple_pay_or_google_pay |
| 3 | atm_support |
| 4 | automatic_payments |
| 5 | balance_not_updated_after_bank_transfer |
| 6 | balance_not_updated_after_cheque_or_cash_deposit |
| 7 | beneficiary_not_allowed |
| 8 | cancel_transfer |
| 9 | card_about_to_expire |
| 10 | card_acceptance |
| 11 | card_activation_required |
| 12 | card_not_received |

### Credit Cards (IDs 13-23)
| ID | Intent Name |
|----|-------------|
| 13 | card_payment_not_recognized |
| 14 | card_payment_wrong_exchange_rate |
| 15 | card_rewards |
| 16 | change_pin |
| 17 | compromised_card |
| 18 | contactless_not_working |
| 19 | country_support |
| 20 | declined_card_payment |
| 21 | declined_cash_withdrawal |
| 22 | declined_transfer |
| 23 | direct_debit_payment_not_recognized |

### Kitchen & Dining (IDs 24-29)
| ID | Intent Name |
|----|-------------|
| 24 | dining_airline_food_status |
| 25 | dining_buffet |
| 26 | dining_calories |
| 27 | dining_complaint |
| 28 | dining_faq |
| 29 | dining_food_last_seen |

### Home (IDs 30-38)
| ID | Intent Name |
|----|-------------|
| 30 | home_appliance |
| 31 | home_automation |
| 32 | home_bed_not_smart |
| 33 | home_brightness |
| 34 | home_cleaning |
| 35 | home_control |
| 36 | home_devices |
| 37 | home_garden |
| 38 | home_getting_started |

### Auto & Commute (IDs 39-47)
| ID | Intent Name |
|----|-------------|
| 39 | auto_accidents |
| 40 | auto_age |
| 41 | auto_insurance |
| 42 | auto_make |
| 43 | auto_mileage |
| 44 | auto_new_car |
| 45 | auto_registration_expired |
| 46 | auto_rental |
| 47 | auto_repair |

### Travel (IDs 48-62)
| ID | Intent Name |
|----|-------------|
| 48 | travel_access_ticket |
| 49 | travel_alert |
| 50 | travel_amenity |
| 51 | travel_baggage |
| 52 | travel_booking_changes |
| 53 | travel_cancel |
| 54 | travel_carry_on |
| 55 | travel_complaint |
| 56 | travel_complimentary_beverage |
| 57 | travel_directions |
| 58 | travel_emergency_flight |
| 59 | travel_enquire |
| 60 | travel_exchange |
| 61 | travel_first_class_question |
| 62 | travel_flight_status |

### Utility (IDs 63-79)
| ID | Intent Name |
|----|-------------|
| 63 | utility_activate_service |
| 64 | utility_bill_balance |
| 65 | utility_bill_explanation |
| 66 | utility_bill_general |
| 67 | utility_bill_late |
| 68 | utility_bill_not_received |
| 69 | utility_cancel_service |
| 70 | utility_credit_limit |
| 71 | utility_customer_support |
| 72 | utility_date_format |
| 73 | utility_different_city |
| 74 | utility_disconnect_or_cancel |
| 75 | utility_downgrade_service |
| 76 | utility_estimate |
| 77 | utility_general_cancel |
| 78 | utility_holiday_mode |
| 79 | utility_new_connection |

### Work (IDs 80-90)
| ID | Intent Name |
|----|-------------|
| 80 | work_accept_contract |
| 81 | work_accept_offer |
| 82 | work_pto_balance |
| 83 | work_pto_request_cancel |
| 84 | work_pto_request_deny |
| 85 | work_pto_used |
| 86 | work_schedule_meeting |
| 87 | work_pto_balance_no_show |
| 88 | work_schedule_sync_ups |
| 89 | work_transfer |
| 90 | work_pto_request |

### Small Talk (IDs 91-100)
| ID | Intent Name |
|----|-------------|
| 91 | small_talk_affirmation |
| 92 | small_talk_bad_time_for_meeting |
| 93 | small_talk_calendar_update |
| 94 | small_talk_change_user_setting |
| 95 | small_talk_chitchat |
| 96 | small_talk_complain_pc |
| 97 | small_talk_eat |
| 98 | small_talk_favor |
| 99 | small_talk_good_morning |
| 100 | small_talk_goodbye |

### Meta (IDs 101-149)
| ID | Intent Name |
|----|-------------|
| 101 | meta_help |
| 102 | meta_how_can_i_help |
| 103 | meta_language_select |
| 104 | meta_no |
| 105 | meta_no_preference |
| 106 | meta_out_of_scope |
| 107 | meta_request_search |
| 108 | meta_reset |
| 109 | meta_restart |
| 110 | meta_thank_you |
| 111 | meta_yes |
| 112 | banking_related |
| 113 | credit_cards |
| 114 | dining |
| 115 | home |
| 116 | auto |
| 117 | travel |
| 118 | utility |
| 119 | work |
| 120 | small_talk |
| 121 | out_of_scope |
| 122 | duplicate_transaction |
| 123 | exchange_rate |
| 124 | high_value_payment |
| 125 | lost_or_stolen_card |
| 126 | new_card_not_received |
| 127 | pending_cash_withdrawal |
| 128 | pending_card_payment |
| 129 | pending_top_up |
| 130 | pending_transfer |
| 131 | pin_usage_consent |
| 132 | receiving_money |
| 133 | Refund_not_showing_up |
| 134 | request_coin_exchange |
| 135 | reverted_card_payment |
| 136 | supported_cards_and_currencies |
| 137 | terminate_change_pin |
| 138 | top_up_declined |
| 139 | top_up_failed |
| 140 | top_up_limits |
| 141 | top_up_not_recognized |
| 142 | transfer_fee |
| 143 | transfer_into_account |
| 144 | transfer_not_received |
| 145 | transfer_timing |
| 146 | unable_to_verify_identity |
| 147 | verify_my_identity |
| 148 | verify_source_of_funds |
| 149 | verify_top_up |

## Example: Intent 83

**Intent ID**: 83  
**Intent Name**: `work_pto_request_cancel`  
**Domain**: Work  
**Description**: User wants to cancel a paid time off (PTO) request

### Example Queries
- "Cancel my time off request"
- "I need to cancel my PTO"
- "Remove my vacation request"

## Model Performance by Intent

See `classification_report.txt` for detailed per-intent precision, recall, and F1 scores.

### Overall Metrics
- **Accuracy**: 95.4%
- **Macro F1**: 0.954
- **Weighted F1**: 0.954
- **Macro Precision**: 0.956
- **Macro Recall**: 0.954

### Confidence Analysis
- **Average Confidence**: 0.952
- **High Confidence Ratio** (>0.7): 95%
- **High Confidence Accuracy**: 97.8%
- **Low Confidence Accuracy**: 50.2%

## Usage

To get the intent name for a prediction:

```python
import json

with open("results/intent_mapping.json") as f:
    mapping = json.load(f)

intent_id = 83
intent_name = mapping["intents"][str(intent_id)]
print(f"Intent {intent_id}: {intent_name}")
```

## References

- **Dataset**: [CLINC150 on HuggingFace](https://huggingface.co/datasets/DeepPavlov/clinc150)
- **Paper**: [CLINC: A Benchmark for Intent Detection](https://www.aclweb.org/anthology/D19-1131.pdf)
- **Model**: DistilBERT (150 intent classes)
