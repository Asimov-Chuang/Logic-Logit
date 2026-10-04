"""Teaching implementation adapted from Shuhan Zhang's Logic-Logit notebook.

The original nested conditional-gradient / column-generation structure is retained.
Support fitting is a heuristic Adam solve, not an exact linear minimization oracle.
See README.md for the scope and changes relative to the supplied tutorial.
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader

# define the sales class to store the information of the dataset
class Sales:
    def __init__(self, p_offerset, p_sell, p_mask):
        # N_sales (num_offers, num_products)
        self.N_sales = torch.tensor(p_sell, dtype=torch.float32)
        # offerset (num_offers, num_products, num_features)
        self.offerset = torch.tensor(p_offerset, dtype=torch.float32)
        # mask (num_offers, num_products)
        self.mask = torch.tensor(p_mask, dtype=torch.float32)
        # Some key information
        self.num_offers = self.offerset.shape[0]
        self.num_products = self.offerset.shape[1]
        self.num_features = self.offerset.shape[2]
        if self.offerset.ndim != 3 or self.N_sales.shape != self.offerset.shape[:2] or self.mask.shape != self.N_sales.shape:
            raise ValueError("Expected X=(sessions, alternatives, predicates) and matching y/mask.")
        if not torch.isfinite(self.offerset).all() or not torch.all((self.offerset == 0) | (self.offerset == 1)):
            raise ValueError("Predicates must be finite binary values.")
        if not torch.all((self.mask == 0) | (self.mask == 1)) or not torch.all(self.mask.sum(1) > 0):
            raise ValueError("Each offer must have a nonempty binary mask.")
        if not torch.all((self.N_sales == 0) | (self.N_sales == 1)) or not torch.all(self.N_sales.sum(1) == 1) or torch.any(self.N_sales > self.mask):
            raise ValueError("Each session must contain exactly one chosen, valid alternative.")
        self.N = torch.sum(self.N_sales)
        self.predicate_list = list(range(self.num_features))

    def cal_rule_based_sales(self, rule_set):
        # create an all-ones tensor to store the rule based features
        rule_based_sales = torch.ones((self.num_offers, self.num_products, len(rule_set)))
        for i, conjunction in enumerate(rule_set):
            for predicate in conjunction:
                rule_based_sales[: ,:, i] = rule_based_sales[: ,:, i] * self.offerset[:, :, predicate]
        return rule_based_sales

class Problem_FrankWolfe:
    def __init__(self, p_offerset, p_sell, p_mask,
                 initial_rule_num = 10,
                 max_rule_length = 2,
                 rule_entering_num = 3,
                 p_product_feature = None,
                 p_customer_feature = None,
                 p_column_generation_num = 50,
                 initial_epochs=100, support_epochs=100, mixture_epochs=500, verbose=False):
        self.initial_epochs = initial_epochs
        self.support_epochs = support_epochs
        self.mixture_epochs = mixture_epochs
        self.verbose = verbose
        if min(initial_epochs, support_epochs, mixture_epochs, initial_rule_num, rule_entering_num, p_column_generation_num) < 1:
            raise ValueError("Epoch counts and rule-search sizes must be positive.")
        if max_rule_length < 1:
            raise ValueError("max_rule_length must be positive.")
        # Define a dataset and sales data
        self.sales = Sales(p_offerset, p_sell, p_mask)
        self.dataset = CustomDataset(self.sales.offerset, self.sales.N_sales, self.sales.mask)
        self.train_loader = DataLoader(self.dataset, batch_size=1024, shuffle=True)

        # Define the main problem NLL loss
        self.NLL_main = None

        # Define the current likelihood convex combination
        self.g = None

        # Define the current likelihood gradient for support finding
        self.NLL_gradient = None

        # Define a list to contain all choice likelihood
        self.fw_list = []

        # Define a list to contain all taste vector
        self.taste_list = []

        # Define a list to contain all proportion
        self.proportion = [1]

        # Define the maximum of rule number in a preference type
        p_product_feature = list(range(self.sales.num_features)) if p_product_feature is None else list(p_product_feature)
        p_customer_feature = [] if p_customer_feature is None else list(p_customer_feature)
        if not p_product_feature:
            raise ValueError("At least one product predicate is required.")
        all_indices = p_product_feature + p_customer_feature
        if len(set(all_indices)) != len(all_indices) or any(i < 0 or i >= self.sales.num_features for i in all_indices):
            raise ValueError("Product/customer predicate indices must be disjoint, unique, and in range.")
        self.initial_rule_num = min(initial_rule_num, len(p_product_feature))
        self.max_rule_num = self.initial_rule_num

        # Define the product and customer features
        self.product_feature = p_product_feature
        self.customer_feature = p_customer_feature
        self.product_predicates = [{i} for i in p_product_feature]
        self.customer_predicates = [{i} for i in p_customer_feature]

        # Define the maximum rule length
        self.max_rule_length = max_rule_length

        # Define column_generation_num
        self.column_generation_num = p_column_generation_num


        self.rule_entering_num = rule_entering_num

        # Define the rule sets of each preference types
        self.rule_set_list = [[{i} for i in range(self.sales.num_features)]]

    def initialize(self):
        # We intialize with training a single MNL Model on all the predicates
        initial_preference = Preference(self.sales.num_features)

        self._log('Initial Training Begin')
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(initial_preference.parameters(), lr=1e-2)
        num_epochs = self.initial_epochs
        for epoch in range(num_epochs):
            for batch_idx, (offerset, sell, mask) in enumerate(self.train_loader):
                log_choice_p = initial_preference(offerset, mask)
                sell = sell.type(torch.int64).argmax(dim=1)
                loss = criterion(log_choice_p, sell)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                log_choice_p = initial_preference(self.sales.offerset, self.sales.mask)
                loss = criterion(log_choice_p, self.sales.N_sales.type(torch.int64).argmax(dim=1))

            if (epoch + 1) % 10 == 0:
                self._log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        self._log('Initial Training End')

        # Add the initial consumer type in the consumer list
        with torch.no_grad():
            self.taste_list.append(initial_preference)
            self.fw_list.append(torch.exp(log_choice_p))
        # Update the main problem NLL loss
        self.main_problem_loss()


    def _log(self, *args):
        if self.verbose:
            print(*args)

    def main_problem_loss(self):
        with torch.no_grad():
            f = sum(weight * probabilities for weight, probabilities in zip(self.proportion, self.fw_list))
            chosen = self.sales.N_sales.argmax(dim=1)
            rows = torch.arange(self.sales.num_offers)
            chosen_p = f[rows, chosen].clamp_min(1e-12)
            self.NLL_main = -chosen_p.log().mean()
            self.NLL_gradient = torch.zeros_like(f)
            self.NLL_gradient[rows, chosen] = -1.0 / (self.sales.num_offers * chosen_p)
        return self.NLL_main

    def support_finding_loss(self, p_choice_p, p_gradient):
        pre_loss = p_gradient * p_choice_p
        pre_loss = torch.where(torch.isnan(pre_loss), torch.tensor(0.0), pre_loss)
        return torch.sum(pre_loss)


    def proportion_update_loss(self, alpha):
        alpha = F.softmax(alpha, dim=0)
        N_sales = self.sales.N_sales
        fw_tensor = torch.stack(self.fw_list, dim=0)
        chosen_indices = N_sales.type(torch.int64).argmax(dim=1)
        chosen_probs = fw_tensor[:, torch.arange(fw_tensor.shape[1]), chosen_indices]
        mixed_probs = torch.sum(chosen_probs * alpha, dim=0)
        log_probs = torch.log(mixed_probs + 1e-10)
        loss = -torch.sum(log_probs) / self.sales.num_offers
        return loss



    def fit_preference(self, p_preference, p_offerset, p_mask, lr=5e-2, num_epochs=None):
        num_epochs = self.support_epochs if num_epochs is None else num_epochs
        criterion = self.support_finding_loss
        optimizer = optim.Adam(p_preference.parameters(), lr=lr)
        dataset = CustomDataset(p_offerset, self.NLL_gradient, p_mask)
        train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
        # Training loop
        for epoch in range(num_epochs):
            for batch_idx, (offerset_batch, gradient_batch, mask_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                log_choice_p = p_preference(offerset_batch, mask_batch)
                choice_p = torch.exp(log_choice_p)
                loss = criterion(choice_p, gradient_batch)  # Assuming gradient_batch is available
                loss.backward()
                optimizer.step()

            # Print loss every 5 epochs
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    log_choice_p = p_preference(p_offerset, p_mask)
                    choice_p = torch.exp(log_choice_p)
                    loss = criterion(choice_p, self.NLL_gradient)
                self._log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Return choice probabilities and the fitted preference model
        with torch.no_grad():
            choice_p = torch.exp(p_preference(p_offerset, p_mask))
        return choice_p, p_preference, loss.item()


    def support_finding(self):
        self._log('-----Consumer Type Search Begin-----')
        # We can first search predicates according to weights, which will make the searching space much smaller

        # Only product predicates are useful in the first fitting
        self._log('Predicate initial fitting')
        p_offerset = self.sales.offerset[:, :, self.product_feature]
        new_preference = Preference(len(self.product_feature))
        choice_p, new_preference, _ = self.fit_preference(new_preference, p_offerset, self.sales.mask)

        # Extract the dominating predicates
        self._log('Predicate extract and refit')
        abs_weight_tensor_fitted = torch.abs(new_preference.linear.weight.data)
        _, indices = torch.topk(abs_weight_tensor_fitted, self.max_rule_num)
        indices = indices.flatten().tolist()
        predicate_set = [self.product_feature[idx] for idx in indices]

        # Refit with the dominating predicates
        p_offerset = self.sales.offerset[:, :, predicate_set]
        new_preference = Preference(self.max_rule_num)
        choice_p, new_preference, current_loss = self.fit_preference(new_preference, p_offerset, self.sales.mask)
        with torch.no_grad():
            current_weight = new_preference.linear.weight.data

        # Next we begin our exciting column generation rule search!
        # Search conjunctions up to the configured maximum rule length.
        # 1. conjunction within the predicate set
        # 2. conjunction between the predicate set with customer predicates

        current_rule_set = [{i} for i in predicate_set]
        for length in range(2, self.max_rule_length + 1):
            self._log('Rule Length:', length)
            # Candidate rules generation
            new_rule_set_1 = self.self_rule_generation(current_rule_set, length)
            new_rule_set_2 = self.conditional_rule_generation(current_rule_set, self.customer_predicates, length)
            new_rule_set = new_rule_set_1 + new_rule_set_2
            # Column Generation
            batch_size = self.column_generation_num
            num_batches = len(new_rule_set) // batch_size + (len(new_rule_set) % batch_size > 0)
            for i in range(num_batches):
                self._log('Column Generation', i + 1, '/', num_batches)
                # Calculate the rule potentials
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(new_rule_set))
                batch_rules = new_rule_set[start_idx:end_idx]
                if not batch_rules:
                    self._log('No enough rules')
                    continue
                potentials = torch.abs(self.rule_potential_calculation(current_rule_set, batch_rules, current_weight))
                _, indices = torch.topk(potentials, min(self.rule_entering_num, len(batch_rules)))
                indices = indices.flatten().tolist()
                potential_rules = [batch_rules[idx] for idx in indices if potentials.flatten()[idx] > 0]
                if not potential_rules:
                    continue
                self._log('Rule Extending')
                # Rule set extending according to potentials
                extended_rule_set = current_rule_set + potential_rules
                p_offerset = self.sales.cal_rule_based_sales(extended_rule_set)
                new_preference = Preference(len(extended_rule_set))
                with torch.no_grad():
                    new_preference.linear.weight.data[:, :self.max_rule_num] = current_weight
                    new_preference.linear.weight.data[:, self.max_rule_num:] = 0
                choice_p, new_preference, new_loss = self.fit_preference(new_preference, p_offerset, self.sales.mask)
                # Check descent if needed
                current_rule_set = extended_rule_set
                self.max_rule_num = len(extended_rule_set)
                # Rule Prune if needed
                with torch.no_grad():
                    current_weight = new_preference.linear.weight.data

        self.max_rule_num = self.initial_rule_num
        with torch.no_grad():
            self.taste_list.append(new_preference)
            self.fw_list.append(choice_p)
            self.rule_set_list.append(current_rule_set)
        self._log('-----Consumer Type Search End-----')

    def self_rule_generation(self, p_basis, p_length):
        candiate_rule_set = []
        new_rules_list = list(itertools.combinations(p_basis, 2))
        for rule in new_rules_list:
            new_rule = rule[0] | rule[1]
            if new_rule not in candiate_rule_set and new_rule not in p_basis and len(new_rule) == p_length:
                candiate_rule_set.append(new_rule)
        return candiate_rule_set


    def conditional_rule_generation(self, p_basis_in, p_basis_out, p_length):
        p_basis_in = [rule for rule in p_basis_in if len(rule) == p_length - 1]
        candiate_rule_set = []
        for rule_in in p_basis_in:
            for rule_out in p_basis_out:
                new_rule = rule_in | rule_out
                if new_rule not in candiate_rule_set:
                    candiate_rule_set.append(new_rule)
        return candiate_rule_set


    def rule_potential_calculation(self, p_current_rule_set,
                                   p_candidate_rule_set, p_current_weight):
        # To calculate the potential of the candidate rules we need a linear layer
        # We set the candidate rule weights initially to 0
        # in order to see their potentials to deviate
        dummy_rule_set = p_current_rule_set + p_candidate_rule_set
        dummy_preference = Preference(len(dummy_rule_set))
        with torch.no_grad():
            dummy_preference.linear.weight.data[:, :self.max_rule_num] = p_current_weight
            dummy_preference.linear.weight.data[:, self.max_rule_num:] = 0

        choice_p = torch.exp(dummy_preference(self.sales.cal_rule_based_sales(dummy_rule_set), self.sales.mask))
        loss = self.support_finding_loss(choice_p, self.NLL_gradient)
        loss.backward()

        with torch.no_grad():
            gradients_on_zeros = dummy_preference.linear.weight.grad.data[:, self.max_rule_num:]

        return gradients_on_zeros




    def proportion_update(self):
        self._log('-----Proportion Update Search Begin-----')
        alpha = torch.empty((len(self.taste_list), 1), dtype=torch.float32, requires_grad=True)
        init.uniform_(alpha, 0, 0)
        optimizer = optim.Adam([alpha], lr=5e-2)

        epoches = self.mixture_epochs
        for epoch in range(epoches):
            loss = self.proportion_update_loss(alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                self._log(f'Epoch [{epoch + 1}/{epoches}], Loss: {loss.item():.4f}')
        old_proportion = self.proportion + [0.0] * (len(self.fw_list) - len(self.proportion))
        old_loss = self.NLL_main.item()
        self.proportion = F.softmax(alpha, dim=0).flatten().tolist()
        self.main_problem_loss()
        # An approximate corrective solve should never make the incumbent worse.
        if not torch.isfinite(self.NLL_main) or self.NLL_main.item() > old_loss:
            self.proportion = old_proportion
            self.main_problem_loss()
        self._log(self.proportion)
        self._log('-----Proportion Update Search End-----')


    def predict_proba(self, offerset, mask):
        """Predict with the fitted predicate order, including variable offer sizes."""
        x = torch.as_tensor(offerset, dtype=torch.float32)
        m = torch.as_tensor(mask, dtype=torch.float32)
        if not self.taste_list:
            raise ValueError("Initialize and train the model before prediction.")
        if x.ndim != 3 or x.shape[-1] != self.sales.num_features or m.shape != x.shape[:2]:
            raise ValueError("Prediction feature or mask shape does not match training.")
        if not torch.isfinite(x).all() or not torch.all((m == 0) | (m == 1)) or not torch.all(m.sum(1) > 0):
            raise ValueError("Predictions require finite features and nonempty binary masks.")
        with torch.no_grad():
            probabilities = torch.zeros(m.shape)
            for weight, preference, rules in zip(self.proportion, self.taste_list, self.rule_set_list):
                features = torch.stack([x[..., sorted(rule)].prod(-1) for rule in rules], dim=-1)
                probabilities += weight * preference(features, m).exp()
        return probabilities.numpy()


class Preference(nn.Module):
    def __init__(self, p_num_feature):
        super(Preference, self).__init__()
        self.linear = nn.Linear(p_num_feature, 1)



    def forward(self, p_offerset, p_mask):
        output = self.linear(p_offerset).squeeze(-1)
        masked_e = torch.where(p_mask == 1, output, float('-inf'))
        log_choice_p = F.log_softmax(masked_e, dim=-1)
        return log_choice_p


class CustomDataset(Dataset):
    def __init__(self, offerset_tensor, sell_tensor, mask_tensor):
        self.offerset_tensor = offerset_tensor
        self.sell_tensor = sell_tensor
        self.mask_tensor = mask_tensor

    def __len__(self):
        return len(self.offerset_tensor)

    def __getitem__(self, idx):
        return self.offerset_tensor[idx], self.sell_tensor[idx], self.mask_tensor[idx]

def train_frank_wolfe(type_num, offerset, sell, mask, **kwargs):
    """Fit the initial component, then add `type_num` candidate taste components."""
    if type_num < 0:
        raise ValueError("type_num must be nonnegative.")
    problem = Problem_FrankWolfe(offerset, sell, mask, **kwargs)
    problem.initialize()
    history = [problem.NLL_main.item()]
    for _ in range(type_num):
        problem.support_finding()
        problem.proportion_update()
        history.append(problem.NLL_main.item())
    return problem, history


def choice_nll(probabilities, choices):
    """Mean negative log likelihood of the observed alternatives."""
    p = np.asarray(probabilities)
    y = np.asarray(choices)
    if p.shape != y.shape or not np.all(y.sum(1) == 1):
        raise ValueError("Expected matching probability and one-hot choice arrays.")
    return float(-np.log(np.maximum(p[np.arange(len(y)), y.argmax(1)], 1e-12)).mean())
