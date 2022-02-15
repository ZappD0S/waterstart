# FX Trading Bot with Policy Gradients

This project explores the development of a trading bot for the FX (Foreign Exchange) market
using policy gradients and reinforcement learning. The goal is to learn a trading strategy
that maximizes returns while managing risk.


## Overview

This project utilizes an [actor-critic](https://en.wikipedia.org/wiki/Actor-critic_algorithm) model to learn a trading strategy. The core of the model is a RNN that processes market state (historical prices) and account state (margin, unrealized profits/losses) to determine optimal actions.

## Architecture

The model architecture is composed of the following components:

1.  **Market State Processing:**

    A rolling window of recent price data, spanning a fixed time duration, is fed into a 1D ConvNet.
    Ideally, this allows the model to extract meaningful features that capture market trends within this time frame.

2.  **Account State Processing:**

    An array containing the currently used margins and unrealized profits/losses for each FX pair is processed by a separate MLP network. These values are normalized dividing by the current account balance.

3.  **Combined Input to RNN:**

    The feature vectors from the ConvNet and the account state network are combined and provided as input to the RNN, along with the hidden state from the previous time step. The RNN is intended to help the model retain memory of observed market patterns and past actions, allowing it to to execute a coherent long-term trading strategy.

4.  **Action Prediction:**

    The output of the RNN is passed through a fully-connected layer to predict the target margins to be allocated for each FX pair. This is achieved by parameterizing a [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution).

1.  **Environment Interaction:**

    The FX trading environment uses the predicted target margins to open or close positions based on the current account state (i.e. open positions).

2. **Critic Input**:

   A critic network (implemented as another MLP) receives the same feature vectors as the RNN and outputs a state-value estimate $V(s)$, which is used within the actor-critic framework.


## Loss evaluation and training

Training is performed on batches of historical data windows, which are randomly selected but with the constraint that they should not overlap. Each window contains price data sampled at 30-minute intervals for the traded FX pairs. For each window, trading is simulated using the model's policy, and the corresponding rewards are calculated.

The reward function is defined using logarithmic returns when a position is closed:

```math
r_i = \log \left(\frac{\text{new\_balance}}{\text{balance}}\right) = \log \left(\frac{\text{balance} + \text{profit\_loss}}{\text{balance}}\right)
```

At all other time steps where no position is closed, the reward is zero.

The advantage estimate is derived from the **temporal-difference (TD) error**:

$$\delta_i = r_i + \gamma V(s_{i+1}) - V(s_i)$$

where:
- $r_i$ is the reward at time step $i$,
- $V(s_i)$ and $V(s_{i+1})$ are the critic’s estimated state values for the current and next state, respectively,
- $\gamma$ is the discount factor.

Using this, the **actor loss** (policy loss) is defined as:

$$
L_{\text{actor}} = \mathop{\mathbb{E}}\limits_{\tau \sim \pi_\theta} \left[ -\log \pi_\theta(a_i | s_i) \cdot \delta_i \right]
$$

Intuitively, this loss encourages the policy $\pi_\theta$ to increase the probability of actions that lead to positive TD errors (higher-than-expected rewards) and decrease the probability of those with negative TD errors.

The **critic loss** (value function loss) is defined as the mean squared TD error:

$$
L_{\text{critic}} = \mathop{\mathbb{E}}\limits_{\tau \sim \pi_\theta} \left[ \frac{1}{2} \delta_i^2 \right]
$$

This updates the critic by minimizing the discrepancy between its value estimates and the observed rewards.

## How to train

At this time, the project is experimental, and the source is provided mainly for reference.
Although the training script is available, it is not very user-friendly at the moment, and I'm not planning to put more work into this. However, if you’re interested in running the training or have questions, please open an issue. With enough interest, I might decide to work on enhancing the usability of the training process.

## Useful resources

- [**Policy Gradient Algorithms** (Lil'log blog)](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- [**Deriving Actor-Critic Method Along with Some Intuition** (Rohan Thorat)](https://rohan-v-thorat.github.io/pages/blogs/Deriving-Actor-Critic-method-along-with-some-intuition.html)
- [**Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients** (Daniel Takeshi)](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/)
- [**Introduction to RL** (OpenAI's Spinning Up)](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [**Reinforcement Learning: An Introduction** (Sutton & Barto, 2nd Edition)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

<!-- - pyro docs -->

