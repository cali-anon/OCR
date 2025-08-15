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
from scipy.special import logit
from scipy.special import perm
from scipy.stats import truncnorm
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from tqdm import tqdm

from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import sigmoid
from obp.utils import softmax
from obp.utils import sample_action_fast
from obp.dataset.base import BaseBanditDataset
from reward_type import RewardType

from sklearn.preprocessing import PolynomialFeatures
from obp.dataset import(
    logistic_reward_function,
    linear_reward_function,
)



def make_allocation_prob(n_category):
    
    values = np.random.uniform(low=0.001, high=10.0, size=(1000, n_category))
    allocation = values / values.sum(axis=1).reshape(-1,1)
    
    return allocation



@dataclass
class SyntheticBanditDataset(BaseBanditDataset):

    n_actions: int
    dim_context: int = 1
    reward_type: str = RewardType.BINARY.value
    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    reward_std: float = 1.0
    action_context: Optional[np.ndarray] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    beta: float = 1.0
    n_deficient_actions: int = 0
    random_state: int = 12345
    n_users: int = 100
    n_category: int = 3
    # true_allocation_prob: Optional[np.ndarray] = None
    lambda_ : float = 0.0
    dataset_name: str = "synthetic_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)
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

        if RewardType(self.reward_type) not in [
            RewardType.BINARY,
            RewardType.CONTINUOUS,
        ]:
            raise ValueError(
                f"`reward_type` must be either '{RewardType.BINARY.value}' or '{RewardType.CONTINUOUS.value}',"
                f"but {self.reward_type} is given.'"
            )
        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)
        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            self.reward_min = 0
            self.reward_max = 1e10

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

    @property
    def len_list(self) -> int:
        """Length of recommendation lists, slate size."""
        return 1

    def sample_contextfree_expected_reward(self) -> np.ndarray:
        """Sample expected reward for each action from the uniform distribution."""
        return self.random_.uniform(size=self.n_actions)

    def calc_expected_reward(self, context: np.ndarray) -> np.ndarray:
        """Sample expected rewards given contexts"""
        # sample reward for each round based on the reward function
        if self.reward_function is None:
            expected_reward_ = np.tile(self.expected_reward, (context.shape[0], 1))
        else:
            expected_reward_ = self.reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )

        return expected_reward_

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Sample reward given expected rewards"""
        expected_reward_factual = expected_reward[np.arange(action.shape[0]), action]
        if RewardType(self.reward_type) == RewardType.BINARY:
            reward = self.random_.binomial(n=1, p=expected_reward_factual)
        elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
            mean = expected_reward_factual
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            reward = truncnorm.rvs(
                a=a,
                b=b,
                loc=mean,
                scale=self.reward_std,
                random_state=self.random_state,
            )
        else:
            raise NotImplementedError

        return reward

    def sample_reward(self, context: np.ndarray, action: np.ndarray) -> np.ndarray:
       
        check_array(array=context, name="context", expected_dim=2)
        check_array(array=action, name="action", expected_dim=1)
        if context.shape[0] != action.shape[0]:
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0]`, but found it False"
            )
        if not np.issubdtype(action.dtype, np.integer):
            raise ValueError("the dtype of action must be a subdtype of int")

        expected_reward_ = self.calc_expected_reward(context)

        return self.sample_reward_given_expected_reward(expected_reward_, action)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
       
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        
        fixed_user_contexts = self.random_.normal(size=(self.n_users, self.dim_context))
        user_idx = self.random_.choice(self.n_users, size=n_rounds)
        contexts = fixed_user_contexts[user_idx]
        
        
        #category
        category = self.random_.choice(self.n_category, size = self.n_actions)
        category[0:self.n_category] = np.arange(self.n_category)


        # calc expected reward given context and action
        fixed_q_x_a = self.calc_expected_reward(fixed_user_contexts)
        expected_reward_ = fixed_q_x_a[user_idx,:]
        if RewardType(self.reward_type) == RewardType.BINARY:
            fixed_q_x_a = sigmoid(fixed_q_x_a)
            expected_reward_ = sigmoid(expected_reward_)
            
            
        # calculate the action choice probabilities of the behavior policy
        if self.behavior_policy_function is None:
            pi_b_logits = fixed_q_x_a
        else:
            pi_b_logits = self.behavior_policy_function(
                context=fixed_user_contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        
        fixed_pi_b = softmax(self.beta * pi_b_logits)
        pi_b = fixed_pi_b[user_idx]

        category_indices = [np.where(category == c)[0] for c in range(self.n_category)]

        fixed_p_x_c = np.zeros((self.n_users, self.n_category))
        for c in range(self.n_category):
            idx = category_indices[c]
            fixed_p_x_c[:,c] = fixed_pi_b[:,idx].sum(axis=1)
            
            
        #allocation
        allocation_candidate_ =  make_allocation_prob(n_category = self.n_category)
        allocation_candidate = np.concatenate([fixed_p_x_c, allocation_candidate_],axis=0)
        allocation_idx_for_user = np.arange(self.n_users) 
        allocation_prob_for_user = allocation_candidate[allocation_idx_for_user]
        
        allocation_idx = allocation_idx_for_user[user_idx]
        allocation_prob = allocation_candidate[allocation_idx]

        idx = self.random_.choice(np.arange(allocation_candidate.shape[0]),size=self.n_users)
        self.true_allocation_prob = allocation_candidate[idx]
        
        
        # sample actions for each round based on the behavior policy
        unique_action = np.arange(self.n_actions)
        actions = np.zeros(n_rounds , dtype=int)
        for i in range(n_rounds):
            actions[i] = self.random_.choice(unique_action, p=pi_b[i]) #sample_action_fast(pi_b, random_state=self.random_state)

        # sample rewards based on the context and action
        rewards = self.sample_reward_given_expected_reward(expected_reward_, actions)
        
        
        temperature = linear_reward_function(
                            context=fixed_user_contexts,
                            action_context=np.eye(1, dtype=int),
                            random_state=self.random_state,
                        )
        temperature = np.abs(temperature)/2
        
        error_array = np.zeros(self.n_users*allocation_candidate.shape[0]).reshape(self.n_users, allocation_candidate.shape[0])
        for i in range(self.n_users):
            error = (self.true_allocation_prob[i]*np.log(self.true_allocation_prob[i]/allocation_candidate)).sum(axis=1)
            error_array[i] = error
        
        V_pi_low_x = np.average(fixed_q_x_a, weights=fixed_pi_b, axis=1).reshape(-1,1)
        
        q_high = (1 - self.lambda_)*np.exp(-temperature*error_array) + self.lambda_*V_pi_low_x  #n_user * len(allocation_candidate)
        q_high_factual = q_high[np.arange(self.n_users), allocation_idx_for_user]
        high_reward = self.random_.binomial(n=1, p=q_high_factual)
        

        return dict(
            n_rounds=n_rounds,
            n_users=self.n_users,
            user_idx=user_idx,
            n_category=self.n_category,
            n_actions=self.n_actions,
            context=contexts,
            fixed_user_context=fixed_user_contexts,
            action_context=self.action_context,
            action=actions,
            position=None,  # position effect is not considered in synthetic data
            reward=rewards,
            high_reward=high_reward,
            expected_reward=expected_reward_,
            fixed_q_x_a=fixed_q_x_a,
            q_high=q_high,
            fixed_pi_b=fixed_pi_b,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
            n_allocation=allocation_candidate.shape[0],
            allocation_idx=allocation_idx,
            allocation_prob=allocation_prob,
            allocation_idx_for_user=allocation_idx_for_user,
            allocation_prob_for_user=allocation_prob_for_user,
            allocation_candidate=allocation_candidate,
            true_allocation_prob=self.true_allocation_prob,
            temperature=temperature,
            category=category,
            category_indices=category_indices,
        )

    
    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
       
        check_array(array=expected_reward, name="expected_reward", expected_dim=2)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if expected_reward.shape[0] != action_dist.shape[0]:
            raise ValueError(
                "Expected `expected_reward.shape[0] = action_dist.shape[0]`, but found it False"
            )
        if expected_reward.shape[1] != action_dist.shape[1]:
            raise ValueError(
                "Expected `expected_reward.shape[1] = action_dist.shape[1]`, but found it False"
            )

        return np.average(expected_reward, weights=action_dist[:, :, 0], axis=1).mean()
