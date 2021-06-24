<!-- The PDF should contain the generated plots, and a short explanation describing the results -->
## Plot of Time per episode vs. episodes and mean entropy per episode vs. episodes
#### Trail example 1 shows relatively good performance
![Time comparison][good]
#### Trail example 2 shows relatively poor performance
![Time comparison][bad]
<!-- descirbe the plots -->
- Plot explanation
  * In the hackathon 8 I trained a reinforcement learning model for game cart pole and testing its performance by comparing the running time and loss respect to the number of episodes run
  * In order to see the evidence phenomenon of longer episodes as experimenting,
  I decide to plot the graph as well as its trend line, the first order derivative is good enough to describe overall whether the function has increasing or decreasing trend.
- Discover
  * As hackathon 8 description, I did notice the increasing trend of time to train per episode, as well as the decreasing mean entropy loss for each episode.
  * As description says, `RL can be very random, so don't be surprised if you get a very good or bad outcome in your first try`, I got a [bad] graph with its mean entropy is fluctuated wildly up and down. But overall trend of the graph is expected.
  * The first [good] trail have significant slope in time per episode vs. mean entropy graph, the slope in the second [bad] trail is not as noticeable as the first [good] trail.
- The policy network and value network both have the same architecture with 2 dense layer with tanh activation
- As function is providing action for given state, it may response with the same state update over time, which might lead to poor performance, as the [paper] mentions, the policy gradient methods is 'using the same trajectory, doing so is not well-justified, and empirically it often leads to destructively large policy updates'.
[good]:hackathon8time_vs_epoch_meanEntropy_vs_epoch.png
[bad]:hackathon8badrun.png
[paper]:https://arxiv.org/abs/1707.06347
