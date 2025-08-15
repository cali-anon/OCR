import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.utils import check_random_state
from tqdm import tqdm
import seaborn as sns
import torch
import random

from sklearn.neural_network import MLPRegressor

from obp.dataset import(
    linear_reward_function,
    linear_behavior_policy,
    logistic_reward_function,
)

from obp.ope import(
    RegressionModel,
)

from policylearners5 import (
    GradientBasedPolicyLearner,
    RegBasedPolicyLearner,
    CalibratedPolicyLearner,
    NetflixCalibratedPolicyLearner,
)

from dataset_real5 import(
    RealBanditDataset,
    make_allocation_prob,
)


num_runs = 50
n_action = 300
epoch = 50
num_data = 60
n_users = 100
n_category = 20
dim_context = 10
random_state = 12345

# random
random.seed(random_state)
# Numpy
np.random.seed(random_state)
# Pytorch
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
torch.backends.cudnn.deterministic = True
torch.manual_seed(random_state)


feedback_influence_list = [0.001,0.01,0.1,1.0] 
tanh_threshold = -0.5

result_df_list_low = []
result_df_list_high = []
result_df_low = DataFrame()
result_df_high = DataFrame()
for feedback_influence in feedback_influence_list:

    fixed_user_context = np.random.normal(size=(19574, dim_context))
    fixed_action_context = torch.from_numpy(np.random.normal(size=(n_action, 7)))
    n_test_data = 2000
    test_dataset = RealBanditDataset(
        dim_context = dim_context,
        n_actions = n_action,
        reward_std = 3.0,
        behavior_policy_function = None,
        beta = 3.0,
        random_state = 12345,
        n_users = n_users,
        n_category = n_category,
        ep_batch_size = n_test_data,
        fixed_user_context=fixed_user_context,
        fixed_action_context=fixed_action_context,
        tanh_threshold=tanh_threshold,
        feedback_influence=feedback_influence,
    )
    
    test_bandit_data = test_dataset.obtain_batch_bandit_feedback(n_rounds = n_test_data)
    
    pi_0_value_low = np.average(test_bandit_data["reward"])
    
    pi_0_value_high = np.average(test_bandit_data["high_reward"])

    dataset = RealBanditDataset(
        dim_context = dim_context,
        n_actions = n_action,
        reward_std = 3.0,
        behavior_policy_function = None,
        beta = 3.0,
        random_state = 12345,
        n_users = n_users,
        n_category = n_category,
        ep_batch_size = num_data,
        fixed_user_context=fixed_user_context,
        fixed_action_context=fixed_action_context,
        tanh_threshold=tanh_threshold,
        feedback_influence=feedback_influence,
    )
    allocation_prob_candidate = make_allocation_prob(dataset.n_category)
    
    test_policy_value_list_high = []
    test_policy_value_list_low = []
    for i in tqdm(range(num_runs), desc=f"feedback_influence={feedback_influence}..."):
        
        test_value_of_learned_policies_high = dict()
        test_value_of_learned_policies_low = dict()
        
        train_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=num_data)
        
        
        train_x = np.concatenate([train_bandit_data["context_for_retention"],train_bandit_data["category_proportion_in_session"]],axis=1)
        reg_model = MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=12345)
        reg_model.fit(train_x,train_bandit_data["high_reward"][:])

        estimated_allocation = np.zeros(num_data, dtype=int)
        for u in range(num_data):
            user_context = np.concatenate([train_bandit_data["context_for_retention"][u].reshape(1,-1)]*allocation_prob_candidate.shape[0],axis=0)
            pre_x = np.concatenate([user_context,allocation_prob_candidate],axis=1)
            predict = reg_model.predict(pre_x)
            estimated_allocation[u] = np.argmin(predict)
            

    
        #IPS
        print("ips--------------------------------------")
        ips = GradientBasedPolicyLearner(dim_context = train_bandit_data["dim_context"], n_action=n_action, epoch = epoch)
        ips.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
        # pi_ips = ips.predict(test_bandit_data)
        ips_value_low, ips_value_high = test_dataset.calc_ground_truth_policy_value(pi=ips, n_rounds=n_test_data)
        
        test_value_of_learned_policies_low["ips"] = ips_value_low
        
        test_value_of_learned_policies_high["ips"] = ips_value_high
        
        
        # calibration_netflix
        print("netflix--------------------------------------")
        
        category = train_bandit_data["category"]
        action = train_bandit_data["action"]
        reward = train_bandit_data["reward"]
        user_idx = train_bandit_data["user_idx"]
        user_number = train_bandit_data["user_number"]
        allocation_prob_estimated = np.zeros((num_data, n_category))

        for i in range(num_data):
            user = 1
            n_watch = (reward*user).sum()
            if n_watch == 0:
                allocation_prob_estimated[i,:] = allocation_prob_candidate[np.random.choice(allocation_prob_candidate.shape[0])]
                continue
            for c in range(n_category):
                action_in_category = np.isin(action, np.where(category == c))
                allocation_prob_estimated[i,c] = (reward * action_in_category*user).sum() / n_watch
                
        cali = NetflixCalibratedPolicyLearner(
            dim_context = train_bandit_data["dim_context"], 
            n_action=n_action, 
            epoch = epoch,
            n_category = dataset.n_category,
            category_indices = train_bandit_data["category_indices"],
            estimated_allocation_prob = torch.from_numpy(allocation_prob_estimated).float(),
        )
        cali.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
        # pi_cali = cali.predict(test_bandit_data)
        cali_value_low, cali_value_high = test_dataset.calc_ground_truth_policy_value(pi=cali, n_rounds=n_test_data)
        
        test_value_of_learned_policies_low["cali"] = cali_value_low
        
        test_value_of_learned_policies_high["cali"] = cali_value_high    
        
        
        
        # ours
        print("ours--------------------------------------")
        
        ours = CalibratedPolicyLearner(
            dim_context = train_bandit_data["dim_context"], 
            n_action=n_action, 
            epoch = epoch,
            n_category = dataset.n_category,
            category_indices = train_bandit_data["category_indices"],
            estimated_allocation_prob = torch.from_numpy(allocation_prob_candidate[estimated_allocation]).float(),
        )
        ours.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
        # pi_ours = ours.predict(test_bandit_data)
        ours_value_low, ours_value_high = test_dataset.calc_ground_truth_policy_value(pi=ours, n_rounds=n_test_data)
        
        test_value_of_learned_policies_low["ours"] = ours_value_low
        
        
        test_value_of_learned_policies_high["ours"] = ours_value_high

        test_policy_value_list_low.append(test_value_of_learned_policies_low)
        test_policy_value_list_high.append(test_value_of_learned_policies_high)
        
        
        
    result_df_low = DataFrame(test_policy_value_list_low).stack().reset_index(1)\
        .rename(columns={"level_1": "method", 0: "value_low"})
    result_df_low["feedback_influence"] = feedback_influence
    result_df_low["pi_0_value_low"] = pi_0_value_low
    result_df_low["rel_value_low"] = result_df_low["value_low"] / pi_0_value_low
    result_df_list_low.append(result_df_low)
    
    result_df_high = DataFrame(test_policy_value_list_high).stack().reset_index(1)\
        .rename(columns={"level_1": "method", 0: "value_high"})
    result_df_high["feedback_influence"] = feedback_influence
    result_df_high["pi_0_value_high"] = pi_0_value_high
    result_df_high["rel_value_high"] = result_df_high["value_high"] / pi_0_value_high
    result_df_list_high.append(result_df_high)

    result_df_low = pd.concat(result_df_list_low).reset_index(level=0)
    result_df_high = pd.concat(result_df_list_high).reset_index(level=0)
    
    result_df_low.to_csv("result_df_low_kuaisim5_val_feedback_influence_seed.csv")
    result_df_high.to_csv("result_df_high_kuaisim5_val_feedback_influence_seed.csv")
    
    
result_df_low = pd.concat(result_df_list_low).reset_index(level=0)
result_df_high = pd.concat(result_df_list_high).reset_index(level=0)

result_df_low.to_csv("result_df_low_kuaisim5_val_feedback_influence_seed.csv")
result_df_high.to_csv("result_df_high_kuaisim5_val_feedback_influence_seed.csv")

