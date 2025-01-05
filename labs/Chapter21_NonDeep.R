#install.packages("ReinforcementLearning")
library(ReinforcementLearning)
library(tidyverse)

set.seed(33)

# Grid size in each dimension (width and height).
grid_size = 5

# Actions the predator can take
actions = c("up", "down", "left", "right")

# Function to move the predator.
# This function accepts a current position (x, y) and an action to take.
# The index of the top row is 1. The index of the left column is 1.
# If we move up from row 2 to 1, we subtract 1.
# If we move left from column 2 to 1, we subtract 1.
# We use min() and max() to avoid going out of bounds.
move = function(position, action) {
  if (action == "up") position[2] = max(1, position[2] - 1)
  if (action == "down") position[2] = min(grid_size, position[2] + 1)
  if (action == "left") position[1] = max(1, position[1] - 1)
  if (action == "right") position[1] = min(grid_size, position[1] + 1)
  return(position)
}

training_episodes = 500

# Initialize the information we will collect as we go.
data = tibble(State = character(), Action = character(), Reward = numeric(), NextState = character())

for (episode in 1:training_episodes) {
  print(paste0("Training Episode: ", episode))

  # Random initial positions for predator and prey
  predator = c(sample(1:grid_size, 1), sample(1:grid_size, 1))
  prey = c(sample(1:grid_size, 1), sample(1:grid_size, 1))
  
  # This variable indicates whether the predator catches the prey in each episode.
  done = FALSE
  
  # This variable keeps track of how many moves have been made per episode.
  # We want to minimize it.
  num_moves = 0

  while (!done) {
    # Indicate that we made another move in this episode.
    num_moves = num_moves + 1
    
    # Current state of the predator and prey. Stored as a concatenated string.
    state = paste(predator[1], predator[2], prey[1], prey[2], sep = "_")
    
    # Randomly select an action
    action = sample(actions, 1)
    
    # Move the predator
    new_predator = move(predator, action)
    
    # Calculate the reward
    if (all(new_predator == prey)) {
      reward = 10 + (10 / num_moves) # Positive reward for catching the prey in relatively few moves
      done = TRUE    # End the episode since the goal is achieved
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
    
    # Create a variable that stores the next state so it can keep track.
    next_state = paste(new_predator[1], new_predator[2], prey[1], prey[2], sep = "_")
    
    # Append the information we gained in this episode to the tibble.
    data = rbind(data, data.frame(State = state, Action = action, Reward = reward, NextState = next_state))
    
    # Update the predator position to the new one.
    predator = new_predator
  }
}

# Train the Q-learning agent
model = ReinforcementLearning(
  data = data,
  s = "State",
  a = "Action",
  r = "Reward",
  s_new = "NextState",
  iter = 100,  # Number of iterations
  control = list(alpha = 0.99, gamma = 0.9, epsilon = 0.1)  # Learning parameters
)

# Examine the learned Q-table
# View(model$Q)

# Now we will sue the trained model to see how well it can guide
#   the movements of predators in new scenarios.
evaluation_episodes = 50
total_steps = 0
successful_catches = 0

for (episode in 1:evaluation_episodes) {
  predator = c(sample(1:grid_size, 1), sample(1:grid_size, 1))
  prey = c(sample(1:grid_size, 1), sample(1:grid_size, 1))
  state = paste(predator[1], predator[2], prey[1], prey[2], sep = "_")
  done = FALSE
  steps = 0
  
  while (!done && steps < 50) {  # Limit steps to prevent infinite loops
    steps = steps + 1
    
    # Choose the best action based on the policy
    action = model$Policy[state]
    if (is.na(action)) {
      action = sample(actions, 1)  # Random action if the state is not in the policy
    }
    
    # Move the predator
    predator = move(predator, action)
    state = paste(predator[1], predator[2], prey[1], prey[2], sep = "_")
    
    # Check if the prey is caught
    if (all(predator == prey)) {
      done = TRUE
    }
  }
  
  total_steps = total_steps + steps
  
  # Output some information for the user to watch things unfold.
  if (done) {
    print(paste0("Episode: ", episode, ", Steps:", steps, ", Result: Prey Caught!"))
    successful_catches = successful_catches + 1
  } else {
    print(paste0("Episode: ", episode, ", Steps:", steps, ", Result: Prey Not Caught!"))
  }
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