"""Train-fitted Expedia-style predicates and a self-contained synthetic demo."""
import numpy as np
import pandas as pd

CONTINUOUS = [
    'position', 'prop_starrating', 'prop_location_score1', 'prop_log_historical_price',
    'srch_booking_window', 'srch_length_of_stay', 'srch_adults_count',
    'srch_children_count', 'srch_room_count', 'price_usd',
]
BINARY = ['prop_brand_bool', 'promotion_flag', 'srch_saturday_night_bool']
QUANTILES = {'prop_location_score1': 10, 'prop_log_historical_price': 10,
             'srch_booking_window': 5, 'price_usd': 10}
FIXED = {
    'position': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40],
    'prop_starrating': [-1, 0, 1, 2, 3, 4, 5],
    'srch_length_of_stay': [0, 1, 2, 3, 4, 5, 10, np.inf],
    'srch_adults_count': [0, 1, 2, 3, 4, np.inf],
    'srch_children_count': [-1, 0, 1, 2, np.inf],
    'srch_room_count': [0, 1, 2, np.inf],
}


class PredicateEncoder:
    """Fit quantile cutpoints on training rows; reuse a fixed feature schema.

    Fixed bins are right-closed. Quantile bins are left-closed, including all
    out-of-training-range values in the extreme bins. Constant columns get one
    quantile bin. Missing or out-of-domain fixed-bin values are rejected.
    """
    def _check(self, frame):
        required = ['srch_id', 'booking_bool'] + CONTINUOUS + BINARY
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        if frame.empty or frame[required].isna().any().any():
            raise ValueError('Data must be nonempty with no missing required values.')
        if not np.isfinite(frame[CONTINUOUS + BINARY + ['booking_bool']].to_numpy(dtype=float)).all():
            raise ValueError('Features and labels must be finite numeric values.')
        if not frame[BINARY + ['booking_bool']].isin([0, 1]).all().all():
            raise ValueError('Binary features and booking_bool must be 0 or 1.')

    def fit(self, frame):
        self._check(frame)
        self.cutpoints_ = {}
        for column, bins in QUANTILES.items():
            edges = np.unique(np.quantile(frame[column], np.linspace(0, 1, bins + 1)))
            self.cutpoints_[column] = edges[1:-1]
        self.feature_names_ = list(BINARY)
        for column in CONTINUOUS:
            if column in QUANTILES:
                edges = [-np.inf, *self.cutpoints_[column], np.inf]
                labels = [f'{column} in [{left:g}, {right:g})' for left, right in zip(edges[:-1], edges[1:])]
            else:
                edges = FIXED[column]
                labels = [f'{column} in {"[" if i == 0 else "("}{left:g}, {right:g}]'
                          for i, (left, right) in enumerate(zip(edges[:-1], edges[1:]))]
            self.feature_names_.extend(labels)
        self.product_indices_ = [i for i, name in enumerate(self.feature_names_) if not name.startswith('srch_')]
        self.customer_indices_ = [i for i, name in enumerate(self.feature_names_) if name.startswith('srch_')]
        # Check fixed-bin domains during fitting too.
        self.transform(frame)
        return self

    def transform(self, frame):
        if not hasattr(self, 'cutpoints_'):
            raise ValueError('Fit on training data before transforming.')
        self._check(frame)
        blocks = [frame[BINARY].to_numpy(dtype=float)]
        for column in CONTINUOUS:
            if column in QUANTILES:
                ids = np.searchsorted(self.cutpoints_[column], frame[column], side='right')
                bins = len(self.cutpoints_[column]) + 1
            else:
                ids = pd.cut(frame[column], FIXED[column], labels=False, include_lowest=True)
                if ids.isna().any():
                    raise ValueError(f'{column} contains values outside the documented fixed bins.')
                ids = ids.to_numpy(dtype=int)
                bins = len(FIXED[column]) - 1
            blocks.append(np.eye(bins, dtype=float)[ids])
        features = pd.DataFrame(np.concatenate(blocks, axis=1), index=frame.index, columns=self.feature_names_)
        return pd.concat([frame[['srch_id', 'booking_bool']].copy(), features], axis=1)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


def pack_offers(frame):
    """Pad variable-size choice sets, preserving predicate order and chosen items."""
    feature_names = [c for c in frame.columns if c not in ('srch_id', 'booking_bool')]
    groups = list(frame.groupby('srch_id', sort=False))
    if not groups:
        raise ValueError('No choice sessions were supplied.')
    width = max(len(g) for _, g in groups)
    x = np.zeros((len(groups), width, len(feature_names)), dtype=np.float32)
    y = np.zeros((len(groups), width), dtype=np.float32)
    mask = np.zeros_like(y)
    for i, (_, group) in enumerate(groups):
        labels = group['booking_bool'].to_numpy()
        if not np.isin(labels, [0, 1]).all() or labels.sum() != 1:
            raise ValueError('Each srch_id must contain exactly one booking.')
        n = len(group)
        x[i, :n] = group[feature_names].to_numpy(dtype=np.float32)
        y[i, :n] = labels
        mask[i, :n] = 1
    return x, y, mask


def make_synthetic_data(n_sessions=200, seed=7):
    """Generate artificial hotel choices; no Expedia records are included.

    Each session has 3-6 alternatives and exactly one sampled booking. A hidden
    two-type preference mixture contains a price/star-rating interaction.
    """
    if n_sessions < 1:
        raise ValueError('n_sessions must be positive.')
    rng = np.random.default_rng(seed)
    rows = []
    for session in range(n_sessions):
        n = int(rng.integers(3, 7))
        customer = {'srch_booking_window': int(rng.integers(1, 80)),
                    'srch_length_of_stay': int(rng.integers(1, 12)),
                    'srch_adults_count': int(rng.integers(1, 5)),
                    'srch_children_count': int(rng.integers(0, 4)),
                    'srch_room_count': int(rng.integers(1, 4)),
                    'srch_saturday_night_bool': int(rng.integers(0, 2))}
        stars = rng.integers(1, 6, n)
        price = rng.uniform(60, 350, n)
        brand = rng.integers(0, 2, n)
        budget_type = rng.random() < 0.55
        utility = (0.35 if budget_type else 1.1) * stars - (0.018 if budget_type else 0.005) * price
        utility += 1.4 * ((stars >= 4) & (price < 180)) + 0.25 * brand
        probs = np.exp(utility - utility.max()); probs /= probs.sum()
        chosen = rng.choice(n, p=probs)
        for j in range(n):
            rows.append(dict(customer, srch_id=session, booking_bool=int(j == chosen),
                             position=j + 1, prop_starrating=stars[j], price_usd=price[j],
                             prop_brand_bool=brand[j], promotion_flag=int(rng.integers(0, 2)),
                             prop_location_score1=float(rng.uniform(0, 6)),
                             prop_log_historical_price=float(np.log(price[j]))))
    return pd.DataFrame(rows)
