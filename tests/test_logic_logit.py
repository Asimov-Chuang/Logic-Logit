import numpy as np
import pytest
import torch
from data_utils import PredicateEncoder, make_synthetic_data, pack_offers
from logic_logit import Sales, train_frank_wolfe, choice_nll


def test_train_test_schema_and_train_only_cutpoints():
    train = make_synthetic_data(12, seed=1)
    test = make_synthetic_data(5, seed=2)
    encoder = PredicateEncoder().fit(train)
    cuts = {k: v.copy() for k, v in encoder.cutpoints_.items()}
    a = encoder.transform(train)
    b = encoder.transform(train.copy())
    np.testing.assert_array_equal(a.values, b.values)
    test['price_usd'] = 100000  # extrapolation must not alter the fitted bins
    c = encoder.transform(test)
    assert list(a.columns) == list(c.columns)
    assert not c.isna().any().any()
    for k in cuts:
        np.testing.assert_array_equal(cuts[k], encoder.cutpoints_[k])
    price_cols = [name for name in encoder.feature_names_ if name.startswith('price_usd in')]
    np.testing.assert_array_equal(c[price_cols].sum(1), 1)
    assert c[price_cols[-1]].eq(1).all()
    # Transform uses names, not the input CSV's column order.
    np.testing.assert_array_equal(c.values, encoder.transform(test[test.columns[::-1]]).values)


def test_constant_quantile_columns_and_unseen_fixed_categories():
    frame = make_synthetic_data(8)
    frame['price_usd'] = 100
    frame['prop_starrating'] = 1
    encoder = PredicateEncoder().fit(frame)
    frame['price_usd'] = 500
    frame['prop_starrating'] = 5
    encoded = encoder.transform(frame)
    assert not encoded.isna().any().any()
    assert encoded['prop_starrating in (4, 5]'].eq(1).all()


def test_invalid_labels_and_domains_are_rejected():
    frame = make_synthetic_data(5)
    encoder = PredicateEncoder().fit(frame)
    bad = frame.copy(); bad['position'] = 100
    with pytest.raises(ValueError, match='outside'):
        encoder.transform(bad)
    encoded = encoder.transform(frame)
    encoded['booking_bool'] = 0
    with pytest.raises(ValueError, match='exactly one'):
        pack_offers(encoded)


@pytest.fixture(scope='module')
def fitted():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    frame = make_synthetic_data(40)
    enc = PredicateEncoder()
    x, y, mask = pack_offers(enc.fit_transform(frame))
    model, history = train_frank_wolfe(
        1, x, y, mask, p_product_feature=enc.product_indices_,
        p_customer_feature=enc.customer_indices_, initial_rule_num=3,
        rule_entering_num=2, initial_epochs=20, support_epochs=15, mixture_epochs=60,
    )
    return enc, model, history, x, y, mask


def test_padding_probabilities_and_cached_training_predictions(fitted):
    enc, model, history, x, y, mask = fitted
    p = model.predict_proba(x, mask)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=2e-6)
    assert (p[mask == 0] == 0).all()
    cached = sum(w * probs for w, probs in zip(model.proportion, model.fw_list)).numpy()
    np.testing.assert_allclose(p, cached, atol=2e-6)
    assert abs(choice_nll(p, y) - history[-1]) < 2e-6
    assert np.all(np.diff(history) <= 1e-7)
    tx, ty, tm = pack_offers(enc.transform(make_synthetic_data(7, seed=55)))
    assert np.isfinite(choice_nll(model.predict_proba(tx, tm), ty))


def test_chosen_probability_gradient(fitted):
    _, model, _, _, y, _ = fitted
    mixture = sum(w * probs for w, probs in zip(model.proportion, model.fw_list)).detach().requires_grad_(True)
    chosen = torch.as_tensor(y).argmax(1)
    expected_loss = -mixture[torch.arange(len(y)), chosen].log().mean()
    expected_loss.backward()
    np.testing.assert_allclose(model.NLL_gradient.numpy(), mixture.grad.numpy(), atol=1e-6)
    assert torch.isfinite(model.NLL_gradient).all()


def test_higher_order_rule_generation_and_feature_alignment(fitted):
    _, model, _, x, y, mask = fitted
    rules = model.self_rule_generation([{0}, {1}, {2}, {0, 1}, {1, 2}], 3)
    assert {0, 1, 2} in rules
    s = Sales(x, y, mask)
    features = s.cal_rule_based_sales([{0, 1}, {2}]).numpy()
    np.testing.assert_array_equal(features[..., 0], x[..., 0] * x[..., 1])
    np.testing.assert_array_equal(features[..., 1], x[..., 2])


def test_invalid_sales_masks(fitted):
    _, _, _, x, y, mask = fitted
    bad = mask.copy(); bad[0] = 0
    with pytest.raises(ValueError, match='nonempty'):
        Sales(x, y, bad)
