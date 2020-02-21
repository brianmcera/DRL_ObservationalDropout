import numpy as np 
import tensorflow as tf

class Agent_Wrapper(object):
    def __init__(self):
        pass

    def _returns_MonteCarlo_advantages(self, rewards, dones, values, next_value):
        # 'next value' is the bootstrapped returned value from the Critic
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma*returns[t+1]*(1-dones[t])
        returns = returns[:-1]  # all but last
        advantages = returns - values  # advantage over Critic estimates
        return returns, advantages

    def _returns_GAE_advantages(self, rewards, dones, values, next_value):
        # 'next value' is the bootstrapped returned value from the Critic
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        values = np.append(values, next_value, axis=-1)
        advantages = np.zeros_like(values)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma*values[t+1]*(1-dones[t]) - values[t]
            advantages[t] = lastgaelam = delta + self.gamma*self.lam*(1-dones[t])*lastgaelam  # double check (1-dones) here
        returns = advantages + values
        returns = returns[:-1]  # all but last
        advantages = advantages[:-1]
        return returns, advantages

    def _value_loss(self, returns_and_prev_values, value):
        # this function calculates loss with value TD error
        returns, prev_values = tf.split(returns_and_prev_values, 2, axis=-1)
        # clip value to reduce variability during Critic training
        vpredclipped = prev_values + tf.clip_by_value(value - prev_values, -self.clip_range, self.clip_range)
        v_losses1 = tf.square(value - returns)
        v_losses2 = tf.square(vpredclipped - returns)
        return self.value_c*tf.reduce_mean(tf.maximum(v_losses1, v_losses2))

    def _logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        probs = tf.nn.softmax(logits)
        # Entropy calculated as crossentropy on itself
        entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        tf.print(entropy_loss, output_stream=sys.stdout)
        return policy_loss - self.entropy_c*entropy_loss

    def _logits_loss_PPO(self, actions_advantages_neglogprobs, logits):
        actions, advantages, neglogprobs_old = tf.split(actions_advantages_neglogprobs, 3, axis=-1)
        neglogprobs_old = tf.squeeze(neglogprobs_old, axis=-1)
        actions = tf.squeeze(tf.cast(actions, tf.int32), axis=-1)
        advantages = tf.squeeze(advantages, axis=-1)
        neglogprobs_new = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
        ratio = tf.exp(neglogprobs_old - neglogprobs_new)  # note order of subtraction, due to *negative* log probabilities
        pg_loss1 = -advantages*ratio
        pg_loss2 = -advantages*tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
        # Calculate policy entropy - entropy calculated as crossentropy on same probability distribution
        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
        return policy_loss - self.entropy_c*entropy_loss
    
    def _normalize_advantages(self, advs, actions, neglogprobs_prev, returns, values, observations):
        # Remove outliers with respect to advantage estimates 
        adv_mean = np.mean(advs)
        adv_std = np.std(advs)
        idx = np.logical_and(advs > adv_mean - 3*adv_std, advs < adv_mean + 3*adv_std)
        advs = advs[idx]
        actions = actions[idx]
        neglogprobs_prev = neglogprobs_prev[idx]
        returns = returns[idx]
        values = values[idx]
        observations = observations[idx]
        return advs, actions, neglogprobs_prev, returns, values, observations
