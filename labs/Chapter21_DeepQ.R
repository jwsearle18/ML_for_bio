library(keras)
library(tensorflow)

##################################################################
# Step 0: Set the random seed
##################################################################

tensorflow::set_random_seed(33)

##################################################################
# Step 1: Define the environment
##################################################################

# The predator and prey interact on a grid. Both have positions
# represented as (x, y) coordinates.

grid_size = 5
actions = c("up", "down", "left", "right")
action_size = length(actions)

# This function accepts a current position (x, y) and an action to take.
# The index of the top row is 1. The index of the left column is 1.
# If we move up from row 2 to 1, we subtract 1.
# If we move left from column 2 to 1, we subtract 1.
# We use min() and max() to avoid going out of bounds.
move = function(position, action) {
  if (action == "up")
    position[2] = max(1, position[2] - 1)
  if (action == "down")
    position[2] = min(grid_size, position[2] + 1)
  if (action == "left")
    position[1] = max(1, position[1] - 1)
  if (action == "right")
    position[1] = min(grid_size, position[1] + 1)

  return(position)
}

##################################################################
# Step 2: Build the Deep Q-Network (DQN)
##################################################################

# The input shape is 4 because the state has 4 values:
#   the x and y positions of the predator
#   the x and y positions of the prey.
# The output shape is 4 because there are 4 actions.
# The number of units is fairly arbitrary, but much lower numbers
#   did not work well in some preliminary tests.
# The output layer uses the linear activation function. We
#   want the Q-value weights to represent expected future rewards
#   for each action. We directly predict the Q-values without
#   constraining them to a specific range.
model = keras_model_sequential() %>%
  layer_dense(units = 24, input_shape = c(4), activation = "relu") %>%
  layer_dense(units = 24, activation = "relu") %>%
  layer_dense(units = action_size, activation = "linear")

# The model is generating continuous outputs, so we use MSE as the loss.
# The choice of the Adam optimizer is arbitrary, but it often performs well.
model %>% compile(
  optimizer = "adam",
  loss = "mse"
)

##################################################################
# Step 3: Train the agent
##################################################################

# Here we specify parameters that will be used in the training process.
# We will use an epsilon-greedy policy. We start the epsilon with
# a large value (the highest it can be) because at first the model will just
# be randomly guessing which direction it should go. Over time, it will
# decay toward the minimum.

episodes = 500             # Number of training episodes
gamma = 0.9               # Discount factor (importance of future rewards)
max_epsilon = 1           # Exploration rate
decay_rate = 0.90         # How much epsilon decays per episode.
min_epsilon = 0.01        # Minimum epsilon after decay.
batch_size = 16           # Number of samples for experience replay

# As we train, we will use this list to record what happens in the training
#   process. It will record what the current state is, what move we took,
#   and what the reward was. We will use this information later to train
#   model. As we do more iterations of training, old information will be
#   lost.
memory = list()           # Experience replay memory

store_experience = function(memory, state, action, reward, next_state, done) {
  # Put the experience information in a list.
  new_experience = list(state, action, reward, next_state, done)
  
  # If the memory is not empty, check whether the last experience is
  # identical to the new one. If so, don't save the new experience.
  # This avoids saving redundant information.
  if (length(memory) > 0) {
    last_experience = memory[[length(memory)]]
    if (identical(last_experience, new_experience)) {
      return(memory)
    }
  }
  
  # Append the new experience to memory (if it was not identical).
  memory[[length(memory) + 1]] = new_experience
  
  # Limit memory size.
  if (length(memory) > 10000)
    memory = memory[-1] #This removes the oldest item.
  
  return(memory)
}

# Epsilon-greedy action selection
choose_action = function(state, current_epsilon) {
  # With probability epsilon, explore (choose a random action).
  if (runif(1) < current_epsilon) {
    return(sample(1:action_size, 1))  # Random action
  } else {
    # Otherwise, exploit the learned policy.
    q_values = model %>% predict(matrix(state, nrow = 1), verbose = 0)
    return(which.max(q_values))      # Find the highest Q-value
  }
}

# Start with the maximum possible epsilon.
epsilon = max_epsilon

# Train the Deep Q-Network
for (episode in 1:episodes) {
  # Reset environment for a new episode
  predator = c(sample(1:grid_size, 1), sample(1:grid_size, 1))  # Random start for predator
  prey = c(sample(1:grid_size, 1), sample(1:grid_size, 1))      # Random start for prey
  state = c(predator, prey)                                     # Initial state
  done = FALSE                                                  # Episode termination flag
  total_reward = 0                                              # Total reward for this episode
  num_moves = 0                                                 # The number of moves we have made per episode
  
  while (!done & num_moves < 50) {
    # Indicate that we made another move in this episode.
    num_moves = num_moves + 1

    action_idx = choose_action(state, epsilon) # Select an action based on the current epsilon.
    action = actions[action_idx] # Map index to the actual action (e.g., "up").

    # Take the action and observe the result.
    # Move the predator based on chosen action.
    new_predator = move(predator, action)

    # Define the new state after both predator and prey have moved.
    new_state = c(new_predator, prey)

    # Calculate the reward
    if (all(new_predator == prey)) {
      reward = 10 + (10 / num_moves) # Positive reward for catching the prey in relatively few moves.
      done = TRUE    # End the episode since the goal is achieved.
    } else {
      # Calculate the Euclidean distance between the predator and prey,
      #   before and after the move. We hope the distance will be smaller
      #   after the move.
      old_distance = sqrt((predator[1] - prey[1])^2 + (predator[2] - prey[2])^2)
      new_distance = sqrt((new_predator[1] - prey[1])^2 + (new_predator[2] - prey[2])^2)

      if (new_distance < old_distance) {
        reward = 1    # Bonus for moving closer
      } else {
        reward = -2   # Penalty for no progress or moving further away
      }
    }

    # Store the experience in memory.
    memory = store_experience(memory, state, action_idx, reward, new_state, done)

    # Update the state and position of the predator.
    predator = new_predator
    state = new_state
    
    # Keep track of the total award.
    total_reward = total_reward + reward
  }

  # Decay epsilon.
  epsilon = max(min_epsilon, epsilon * decay_rate)

  # Print progress for this episode.
  print(paste0("Episode: ", episode, ", Total Reward: ", round(total_reward, 1), ", Num Moves: ", num_moves))

  # Now that we have observed what happens for this episode and the reward,
  # let's see if we have enough observations to train a batch.
  if (length(memory) >= batch_size) {
    # We randomly select experiences from the memory. These will be in
    # random order (on purpose). We don't want to train the model based on
    # specific orderings of what happened in specific episodes. We want
    # to train based on a wide variety of experiences. The experiences
    # we do not train on might be used in a future batch. Or they might
    # be removed from the memory when they get "old."
    batch = sample(memory, batch_size)

    # batch will be a list of lists.
    # We will iterate through the experiences (lists) in the batch.
    for (experience in batch) {
      # States are represented as a vector like this: c(predator_x, predator_y, prey_x, prey_y)
      s = experience[[1]]         # Initial state
      a = experience[[2]]         # Action taken
      r = experience[[3]]         # Reward received
      s_next = experience[[4]]    # New state after action
      d = experience[[5]]         # Whether the episode ended

      # Predict Q-values for the current state.
      # This tells us what the model currently thinks is the best action
      # for the current state.
      # The results will be a matrix with 1 row and action_size columns.
      target = model %>% predict(matrix(s, nrow = 1), verbose = 0)

      # Check if the episode has ended.
      if (d) {
        # If the episode has ended, the Q-value for the chosen action
        # is simply the immediate reward, as there are no future states.
        target[1, a] = r
      } else {
        # If the episode has not ended, predict the Q-values for the next state.
        t_next = model %>% predict(matrix(s_next, nrow = 1), verbose = 0)
        
        # Update the Q-value for the current state and chosen action
        # 'gamma' is the discount factor, which reduces the importance of future rewards.
        target[1, a] = r + gamma * max(t_next)  # Immediate reward + discounted future reward
      }
      
      # Train the model on this experience.
      # The input is the current state, and the target is the updated Q-value
      # for the chosen action. This trains the model to better predict the Q-values
      # for each state-action pair over time.
      model %>% train_on_batch(matrix(s, nrow = 1), target)
    }
  }
}

##################################################################
# Step 4: Evaluate the trained agent
##################################################################

##################################################################
# This script evaluates how well the trained agent performs
# without further training. The agent will always exploit its
# learned policy (chooses the best action based on Q-values).
##################################################################

# Number of evaluation episodes
evaluation_episodes = 50

# Track performance metrics
successful_catches = 0  # Count of episodes where predator catches prey
total_steps = 0         # Total steps taken across all episodes
total_rewards = 0       # Total rewards accumulated across all episodes

for (episode in 1:evaluation_episodes) {
  # Reset the environment for a new episode
  predator = c(sample(1:grid_size, 1), sample(1:grid_size, 1))  # Random start for predator
  prey = c(sample(1:grid_size, 1), sample(1:grid_size, 1))      # Random start for prey
  state = c(predator, prey)                                    # Initial state
  done = FALSE                                                 # Episode termination flag
  episode_steps = 0                                            # Steps in the current episode
  
  # Run the episode until the predator catches the prey or a limit is reached.
  while (!done && episode_steps < 50) {  # Prevent infinite loops with a step limit
    # Use the trained model to choose the best action
    action_idx = choose_action(state, 0)  # Select the best action based on Q-values
    action = actions[action_idx]       # Map the action index to the actual action

    # Move the predator based on the chosen action.
    new_predator = move(predator, action)

    # Check whether the prey was caught.
    if (all(new_predator == prey)) {
      done = TRUE    # End the episode since the goal is achieved
      successful_catches = successful_catches + 1  # Increment success counter
    }

    predator = new_predator          # Update the predator's position
    state = c(predator, prey)        # Update the state with the new positions
    episode_steps = episode_steps + 1  # Increment step counter
  }
  
  # Track cumulative metrics across all episodes.
  total_steps = total_steps + episode_steps
  
  # Print progress for each episode.
  print(paste0("Evaluation Episode: ", episode, ", Steps: ", episode_steps))
}

##################################################################
# Calculate and Display Evaluation Metrics
##################################################################

# Average number of steps per episode
average_steps = total_steps / evaluation_episodes

# Success rate (percentage of episodes where prey was caught)
success_rate = (successful_catches / evaluation_episodes) * 100

# Print final evaluation metrics
print("Evaluation Results:")
print(paste0("Total Episodes Evaluated: ", evaluation_episodes))
print(paste0("Average Steps per Episode: ", average_steps))
print(paste0("Success Rate (%): ", success_rate))