"""Run the synthetic Logic-Logit tutorial as a script."""
import json
import numpy as np
import torch
from data_utils import PredicateEncoder, make_synthetic_data, pack_offers
from logic_logit import train_frank_wolfe, choice_nll


def run_demo():
    np.random.seed(7)
    torch.manual_seed(7)
    torch.set_num_threads(1)
    frame = make_synthetic_data(200, seed=7)
    train = frame[frame.srch_id < 150]
    test = frame[frame.srch_id >= 150]
    encoder = PredicateEncoder()
    tr_x, tr_y, tr_mask = pack_offers(encoder.fit_transform(train))
    te_x, te_y, te_mask = pack_offers(encoder.transform(test))
    model, losses = train_frank_wolfe(
        2, tr_x, tr_y, tr_mask,
        p_product_feature=encoder.product_indices_, p_customer_feature=encoder.customer_indices_,
        initial_rule_num=4, rule_entering_num=2, initial_epochs=40,
        support_epochs=30, mixture_epochs=150,
    )
    probabilities = model.predict_proba(te_x, te_mask)
    summary = {'dataset': 'synthetic (not paper benchmark results)', 'seed': 7,
               'train_sessions': len(tr_x), 'test_sessions': len(te_x),
               'train_nll_history': losses,
               'test_nll': choice_nll(probabilities, te_y),
               'uniform_test_nll': float(np.log(te_mask.sum(1)).mean()),
               'max_probability_sum_error': float(np.abs(probabilities.sum(1) - 1).max()),
               'mixture_weights': model.proportion}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    run_demo()
