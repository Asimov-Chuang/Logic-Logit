# Using processed Expedia data

The default tutorial generates artificial choices. No real Expedia records are
included. The original teaching notebook used processed `train_28-32_10000.csv`
and `test_28-32_1000.csv` files from a hotel-choice dataset.

For your own experiment, prepare `data/train.csv` and `data/test.csv` with:

- Session/label: `srch_id`, `booking_bool`.
- Product: `position`, `prop_starrating`, `prop_location_score1`,
  `prop_log_historical_price`, `price_usd`, `prop_brand_bool`, `promotion_flag`.
- Customer/session: `srch_booking_window`, `srch_length_of_stay`,
  `srch_adults_count`, `srch_children_count`, `srch_room_count`,
  `srch_saturday_night_bool`.

Every session must contain exactly one booking (`booking_bool=1`); other labels
and boolean features must be 0 or 1. Required values must be numeric, finite, and
nonmissing, except session IDs which only need to be nonmissing. Count features
must satisfy the bins in `data_utils.py`; positions are supported from 0 to 40,
and star ratings from 0 to 5. Sessions without a booking or with multiple bookings
need an explicitly chosen preprocessing policy before using this tutorial.

Split by `srch_id`, never by individual alternative rows. Fit `PredicateEncoder`
on training data only, then call its `transform` on validation/test data. The
encoder does not change test columns or use test data to choose bin boundaries.
Quantile edge duplicates are collapsed; constant quantile features have one bin.
Finite values outside the training quantile range enter the outermost bins.

Dataset source referenced by the original notebook:
[Expedia hotel data on Kaggle](https://www.kaggle.com/datasets/vijeetnigam26/expedia-hotel).
Obtain data through its source and follow the source's terms. CSVs are ignored by
Git so local experiments are not accidentally added to this teaching repository.
