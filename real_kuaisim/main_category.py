import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.utils import check_random_state
from tqdm import tqdm
import seaborn as sns
import torch

from sklearn.neural_network import MLPRegressor

from obp.dataset import(
    linear_reward_function,
    logistic_reward_function,
)

from obp.ope import(
    RegressionModel,
)

from policyleaners import (
    GradientBasedPolicyLearner,
    RegBasedPolicyLearner,
    CalibratedPolicyLearner,
    NetflixCalibratedPolicyLearner,
)


from dataset import(
    SyntheticBanditDataset,
    make_allocation_prob,
) 


num_runs = 50
n_action = 300
epoch = 30
num_data = 1000
lambda_ = 0.3
n_users = 300
category_list = [10,20,30,40,50]
random_state = 12345
np.random.seed(12345)




result_df_list_low = []
result_df_list_high = []
result_df_low = DataFrame()
result_df_high = DataFrame()
for n_category in category_list:


    dataset = SyntheticBanditDataset(
        n_actions = n_action,
        dim_context = 5,
        reward_type = "binary",
        reward_function = linear_reward_function,
        reward_std = 3.0,
        behavior_policy_function = None,
        beta = -7.0,
        random_state = 12345,
        n_users = n_users,
        n_category = n_category,
        lambda_ = lambda_,
    )

    test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds = 30000)
    
    
    pi_0_value_low = dataset.calc_ground_truth_policy_value(
                expected_reward=test_bandit_data["expected_reward"], 
                action_dist=test_bandit_data["pi_b"],
            )
    
    
    allocation_prob = np.zeros(dataset.n_users*dataset.n_category).reshape(dataset.n_users, dataset.n_category)
    pi_0_high = test_bandit_data["fixed_pi_b"]
    category_indices = [np.where(test_bandit_data["category"] == c)[0] for c in range(dataset.n_category)]
    for c in range(dataset.n_category):
        idx = category_indices[c]
        allocation_prob[:,c] = pi_0_high[:, idx].sum(axis=1)
    
    V_pi_low_x_0 = np.average(test_bandit_data["fixed_q_x_a"], weights=pi_0_high, axis=1).reshape(-1,1)
    error = (dataset.true_allocation_prob*np.log(dataset.true_allocation_prob/allocation_prob)).sum(axis=1)
    
    pi_0_value_high = np.average((1 - dataset.lambda_)*np.exp(-test_bandit_data["temperature"]*error) + dataset.lambda_*V_pi_low_x_0)



    allocation_prob_candidate = test_bandit_data["allocation_candidate"]
    true_allocation_prob = test_bandit_data["true_allocation_prob"]
    
    test_policy_value_list_high = []
    test_policy_value_list_low = []
    for i in tqdm(range(num_runs), desc=f"n_category={n_category}..."):
        
        test_value_of_learned_policies_high = dict()
        test_value_of_learned_policies_low = dict()
        
        train_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=num_data)


        reg_model = RegressionModel(
                    n_actions = train_bandit_data["n_allocation"],
                    base_model = MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=12345),
                )

        estimated_rewards = reg_model.fit_predict(
            context=train_bandit_data["fixed_user_context"], # context; x
            action=train_bandit_data["allocation_idx_for_user"], # action; a
            reward=train_bandit_data["high_reward"], # reward; r
            random_state=12345,
        )

        estimated_allocation = np.argmax(estimated_rewards, axis = 1)


    
        #IPS
        ips = GradientBasedPolicyLearner(dim_context = 5, n_action=n_action, epoch = epoch)
        ips.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
        pi_ips = ips.predict(test_bandit_data)
        ips_value_low = (test_bandit_data["expected_reward"] * pi_ips).sum(1).mean()
        test_value_of_learned_policies_low["ips"] = ips_value_low
        
        test_policy_value_list_low.append(test_value_of_learned_policies_low)
        
        #obtain \pi(c|x)
        category_indices = test_bandit_data["category_indices"] #[np.where(test_bandit_data["category"] == c)[0] for c in range(dataset.n_category)]
        allocation_prob = np.zeros(dataset.n_users*dataset.n_category).reshape(dataset.n_users, dataset.n_category)
        pi_ips_high = ips.predict_high(test_bandit_data)
        for c in range(dataset.n_category):
            idx = category_indices[c]
            allocation_prob[:,c] = pi_ips_high[:, idx].sum(axis=1)
        
        V_pi_low_x_ips = np.average(test_bandit_data["fixed_q_x_a"], weights=pi_ips_high, axis=1).reshape(-1,1)
        error = (dataset.true_allocation_prob*np.log(dataset.true_allocation_prob/allocation_prob)).sum(axis=1)
        
        ips_value_high = np.average((1 - dataset.lambda_)*np.exp(-test_bandit_data["temperature"]*error) + dataset.lambda_*V_pi_low_x_ips)
        test_value_of_learned_policies_high["ips"] = ips_value_high
        test_policy_value_list_high.append(test_value_of_learned_policies_high)

        
        # calibration_netflix
        
        category = train_bandit_data["category"]
        action = train_bandit_data["action"]
        reward = train_bandit_data["reward"]
        user_idx = train_bandit_data["user_idx"]
        allocation_prob_estimated = np.zeros((n_users, n_category))

        for i in range(n_users):
            user = np.isin(user_idx, i)
            n_watch = (reward*user).sum()
            if n_watch == 0:
                allocation_prob_estimated[i,:] = allocation_prob_candidate[np.random.choice(allocation_prob_candidate.shape[0])]
                continue
            for c in range(n_category):
                action_in_category = np.isin(action, np.where(category == c))
                allocation_prob_estimated[i,c] = (reward * action_in_category*user).sum() / n_watch
                
        cali = NetflixCalibratedPolicyLearner(
            dim_context = 5, 
            n_action=n_action, 
            epoch = epoch,
            n_category = dataset.n_category,
            category_indices = train_bandit_data["category_indices"],
            estimated_allocation_prob = torch.from_numpy(allocation_prob_estimated).float(),
        )
        cali.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
        pi_cali = cali.predict(test_bandit_data)
        cali_value_low = (test_bandit_data["expected_reward"] * pi_cali).sum(1).mean()
        test_value_of_learned_policies_low["cali"] = cali_value_low
        test_policy_value_list_low.append(test_value_of_learned_policies_low)
        
        #obtain \pi(c|x)
        category_indices = test_bandit_data["category_indices"] #[np.where(test_bandit_data["category"] == c)[0] for c in range(dataset.n_category)]
        allocation_prob = np.zeros(dataset.n_users*dataset.n_category).reshape(dataset.n_users, dataset.n_category)
        pi_cali_high = cali.predict_high(test_bandit_data)
        for c in range(dataset.n_category):
            idx = category_indices[c]
            allocation_prob[:,c] = pi_cali_high[:, idx].sum(axis=1)
        
        error = (dataset.true_allocation_prob*np.log(dataset.true_allocation_prob/allocation_prob)).sum(axis=1)
        V_pi_low_x_cali = np.average(test_bandit_data["fixed_q_x_a"], weights=pi_cali_high, axis=1).reshape(-1,1)
        
        cali_value_high = np.average((1 - dataset.lambda_)*np.exp(-test_bandit_data["temperature"]*error) + dataset.lambda_*V_pi_low_x_cali)
        test_value_of_learned_policies_high["cali"] = cali_value_high
        test_policy_value_list_high.append(test_value_of_learned_policies_high)

        
        # ours
        
        ours = CalibratedPolicyLearner(
            dim_context = 5, 
            n_action=n_action, 
            epoch = epoch,
            n_category = dataset.n_category,
            category_indices = train_bandit_data["category_indices"],
            estimated_allocation_prob = torch.from_numpy(allocation_prob_candidate[estimated_allocation[:,0]]).float(),
        )
        ours.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
        pi_ours = ours.predict(test_bandit_data)
        ours_value_low = (test_bandit_data["expected_reward"] * pi_ours).sum(1).mean()
        test_value_of_learned_policies_low["ours"] = ours_value_low
        test_policy_value_list_low.append(test_value_of_learned_policies_low)
        
        pi_ours_high = ours.predict_high(test_bandit_data)
        allocation_prob = np.zeros(dataset.n_users*dataset.n_category).reshape(dataset.n_users, dataset.n_category)
        for c in range(dataset.n_category):
            idx = category_indices[c]
            allocation_prob[:,c] = pi_ours_high[:, idx].sum(axis=1)
        
        error = (dataset.true_allocation_prob*np.log(dataset.true_allocation_prob/allocation_prob)).sum(axis=1)
        V_pi_low_x_ours = np.average(test_bandit_data["fixed_q_x_a"], weights=pi_ours_high, axis=1).reshape(-1,1)
        
        ours_value_high = np.average((1 - dataset.lambda_)*np.exp(-test_bandit_data["temperature"]*error) + dataset.lambda_*V_pi_low_x_ours)
        test_value_of_learned_policies_high["ours"] = ours_value_high
        test_policy_value_list_high.append(test_value_of_learned_policies_high)
        
        
    result_df_low = DataFrame(test_policy_value_list_low).stack().reset_index(1)\
        .rename(columns={"level_1": "method", 0: "value_low"})
    result_df_low["lambda"] = lambda_
    result_df_low["n_category"] = n_category
    result_df_low["pi_0_value_low"] = pi_0_value_low
    result_df_low["rel_value_low"] = result_df_low["value_low"] / pi_0_value_low
    result_df_list_low.append(result_df_low)
    
    result_df_high = DataFrame(test_policy_value_list_high).stack().reset_index(1)\
        .rename(columns={"level_1": "method", 0: "value_high"})
    result_df_high["lambda"] = lambda_
    result_df_high["n_category"] = n_category
    result_df_high["pi_0_value_high"] = pi_0_value_high
    result_df_high["rel_value_high"] = result_df_high["value_high"] / pi_0_value_high
    result_df_list_high.append(result_df_high)

    result_df_low = pd.concat(result_df_list_low).reset_index(level=0)
    result_df_high = pd.concat(result_df_list_high).reset_index(level=0)
    
    result_df_low.to_csv("result_df_low_category_reg_usual_beta7.csv")
    result_df_high.to_csv("result_df_high_category_reg_usual_beta7.csv")
    
    
result_df_low = pd.concat(result_df_list_low).reset_index(level=0)
result_df_high = pd.concat(result_df_list_high).reset_index(level=0)

result_df_low.to_csv("result_df_low_category_reg_usual_beta7.csv")
result_df_high.to_csv("result_df_high_category_reg_usual_beta7.csv")
