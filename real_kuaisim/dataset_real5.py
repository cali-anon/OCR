import obp
from dataclasses import dataclass
from itertools import permutations
from itertools import product
import itertools
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.special import perm
from scipy.stats import truncnorm
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from tqdm import tqdm
import random

from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import sigmoid
from obp.utils import softmax
from obp.utils import sample_action_fast
from obp.dataset.base import BaseBanditDataset
#from reward_type import RewardType

from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from obp.dataset import(
    logistic_reward_function,
    linear_reward_function,
)
from pathlib import Path

#KRL
import torch
from argparse import Namespace

from env.KRCrossSessionEnvironment_GPU import KRCrossSessionEnvironment_GPU
from sklearn.cluster import KMeans



def make_allocation_prob(n_category):

    values = np.random.uniform(low=0.001, high=1.0, size=(1000,n_category))
    
    allocation = values / values.sum(axis=1).reshape(-1,1)
    
    return allocation


seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



@dataclass
class RealBanditDataset(BaseBanditDataset):
    fixed_user_context: np.ndarray=0
    fixed_action_context: np.ndarray=0
    n_actions: int = 300
    dim_context : int = 5
    reward_std: float = 1.0
    action_context: Optional[np.ndarray] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    beta: float = 1.0
    n_deficient_actions: int = 0
    random_state: int = 12345
    n_users: int = 300
    n_category: int = 30
    ep_batch_size: int = 100
    tanh_coef: float = 0.3
    tanh_threshold: float = -0.5
    feedback_influence: float = 0.01
    dataset_name: str = "real_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.beta, "beta", (int, float))
        check_scalar(
            self.n_deficient_actions,
            "n_deficient_actions",
            int,
            min_val=0,
            max_val=self.n_actions - 1,
        )

        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)

        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)
       
        # one-hot encoding characterizing actions.
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)
        else:
            check_array(
                array=self.action_context, name="action_context", expected_dim=2
            )
            if self.action_context.shape[0] != self.n_actions:
                raise ValueError(
                    "Expected `action_context.shape[0] == n_actions`, but found it False."
                )

        self.env = self.pre_process(self.ep_batch_size)
        
    def pre_process(self,ep_batch_size):
        output_path = 'output/Kuairand_Pure/env/log/'
        uirm_log_path = output_path + 'user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log'
        
        slate_size = 1
        max_step = 5
        max_n_session = 4
        max_return_day = 10
        ep_batch_size = ep_batch_size
        initial_temper = max_step
        rho = 0.1
        return_day_bias = 0.3
        # self.feedback_influence = 0.01 #0.01
        args = eval(f"Namespace(uirm_log_path='{uirm_log_path}', slate_size={slate_size}, \
                    max_step_per_episode={max_step}, episode_batch_size={ep_batch_size}, \
                    initial_temper={initial_temper}, item_correlation={rho}, device='cpu', \
                    single_response=False, max_n_session={max_n_session}, \
                    max_return_day={max_return_day}, next_day_return_bias={return_day_bias}, \
                    feedback_influence_on_return={self.feedback_influence})")
        
        env = KRCrossSessionEnvironment_GPU(args)

        return env


    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
       
        check_scalar(n_rounds, "n_rounds", int, min_val=1)

        observation = self.env.reset({'batch_size': self.ep_batch_size})
        self.user_idx = (self.env.current_observation['user_profile']["user_id"]).to('cpu').detach().numpy().copy()

        # fixed_user_contexts = self.random_.normal(size=(self.n_users, self.dim_context))
        # user_idx = self.random_.choice(self.n_users, size=n_rounds)
        # self.user_idx = self.random_.choice(self.n_users, size=n_rounds)
        contexts = self.fixed_user_context[self.user_idx]
        
        
        #category
        # category = self.random_.choice(self.n_category, size = self.n_actions)#self.cluster #self.random_.choice(self.n_category, size = self.n_actions)

        km = KMeans(n_clusters=self.n_category, random_state=self.random_state)
        category = km.fit_predict(self.fixed_action_context)

        #allocation
        allocation_candidate =  make_allocation_prob(n_category = self.n_category)
        allocation_idx = self.random_.choice(allocation_candidate.shape[0], size = n_rounds)
        allocation_prob = allocation_candidate[allocation_idx]
        

        
            
        # calculate the action choice probabilities of the behavior policy
        if self.behavior_policy_function is None:
            pi_b_logits = self.random_.normal(size=(n_rounds, self.n_actions))
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        
        pi_b = softmax(self.beta * pi_b_logits)
        # pi_b = fixed_pi_b[self.user_idx]
            

        category_indices = [np.where(category == c)[0] for c in range(self.n_category)]
        
        
        
        self.env.next_day_return_bias = torch.tensor([0.1]*n_rounds) #0.3
        # self.fixed_action_context = torch.from_numpy(self.fixed_action_context) #torch.randn(self.n_actions, self.env.action_dim)
        action_list = []
        click_list = []
        retention_list = []
        episode_length = 50
        d = torch.tensor([0.0]*n_rounds)
        unique_action_set = np.arange(self.n_actions)
        category_proportion_in_session = np.zeros((n_rounds,self.n_category))
        category_proportion_in_session_list = []
        for i in tqdm(range(episode_length)):
            action_idx = np.zeros(n_rounds, dtype=int)
            for j in range(n_rounds):
                action_idx[j] = self.random_.choice(
                    unique_action_set, p=pi_b[j], replace=False
                )
                category_proportion_in_session[j,category[action_idx[j]]] += 1/5
            action = self.fixed_action_context[action_idx]
            new_observation, user_feedback, updated_observation = self.env.step({'action': action})
            observation = new_observation
            action_list.append(action_idx)
            click_list.append(user_feedback['immediate_response'][:,0,0].numpy())
            if i >0:
                d += np.sqrt(((self.fixed_action_context[action_list[-2]] - action)**2).sum(axis=1)) / 5
            if i%5==4:
                self.env.next_day_return_bias = torch.tanh(d*self.tanh_coef + self.tanh_threshold)
                retention_list.append(user_feedback['retention'].numpy().reshape(-1,1))
                category_proportion_in_session_list.append(category_proportion_in_session)
                d = torch.tensor([0.0]*n_rounds)
                category_proportion_in_session = np.zeros((n_rounds,self.n_category))

        actions = np.concatenate(action_list)
        click = np.concatenate(click_list)
        retention = np.concatenate(retention_list,axis=0) #user_feedback['retention'].numpy()
        category_proportion_in_session = np.concatenate(category_proportion_in_session_list,axis=0)

        user_number = {}
        for i, idx in enumerate(self.user_idx):
            user_number[f"{idx}"] = i

        return dict(
            n_rounds=n_rounds,
            n_tuples=int(n_rounds*episode_length),
            n_users=self.n_users,
            user_idx=np.concatenate([self.user_idx]*episode_length),
            user_number=user_number,
            n_category=self.n_category,
            n_actions=self.n_actions,
            context=np.concatenate([contexts]*episode_length),
            context_for_retention=np.concatenate([contexts]*10),
            fixed_user_context=self.fixed_user_context,
            dim_context=contexts.shape[1],
            action_context=self.action_context,
            action=actions,
            position=None,  # position effect is not considered in synthetic data
            reward=click,
            high_reward=retention,
            pi_b=np.concatenate([pi_b[:, :, np.newaxis]]*episode_length),
            pscore=np.concatenate([pi_b[:, :, np.newaxis]]*episode_length)[np.arange(n_rounds*episode_length), actions],
            n_allocation=allocation_candidate.shape[0],
            allocation_idx=allocation_idx,
            allocation_prob=allocation_prob,
            category=category,
            category_indices=category_indices,
            category_proportion_in_session=category_proportion_in_session,
        )

    
    def calc_ground_truth_policy_value(
        self, pi: np.ndarray, n_rounds,
    ) -> float:

        episode_length = 50
        
        observation = self.env.reset({'batch_size': n_rounds})
        user_idx = (self.env.current_observation['user_profile']["user_id"]).to('cpu').detach().numpy().copy()
        contexts = self.fixed_user_context[self.user_idx]
        bandit_data = {"context": contexts}
        pi_val = pi.predict(bandit_data)

        action_list = []
        click_list = []
        retention_list = []
        # episode_length = 200
        unique_action_set = np.arange(self.n_actions)
        d = torch.tensor([0.0]*n_rounds)
        for i in range(episode_length):
            action_idx = np.zeros(n_rounds, dtype=int)
            for j in range(n_rounds):
                action_idx[j] = self.random_.choice(
                    unique_action_set, p=pi_val[j], replace=False
                )
            action = self.fixed_action_context[action_idx]
            new_observation, user_feedback, updated_observation = self.env.step({'action': action})
            observation = new_observation
            action_list.append(action_idx)
            click_list.append(user_feedback['immediate_response'][:,0,0].numpy())
            # retention_list.append(user_feedback)
            if i>0:
                d += np.sqrt(((self.fixed_action_context[action_list[-2]] - action)**2).sum(axis=1)) / 5
            if i%5==4:
                self.env.next_day_return_bias = torch.tanh(d*self.tanh_coef + self.tanh_threshold)
                retention_list.append(user_feedback['retention'].numpy().reshape(1,-1))
                d = torch.tensor([0.0]*n_rounds)
                

        click = np.concatenate(click_list)
        retention = np.concatenate(retention_list,axis=0).mean(axis=0) 

        

        return np.average(click), np.average(retention)

    def set_(self,n_action,):
        self.n_actions = n_action

