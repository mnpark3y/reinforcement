# Mindy White
# CS 7641: Machine Learning
# Markov Decision Processes
# Large States: Grid World
#
# Initialize Environment
install.packages("MDPtoolbox")
install.packages("ggplot2")
install.packages("reshape2")
require("MDPtoolbox");require(ggplot2);require(reshape2)


# World is a 15x15 grid
x <- 1:15
y <- 1:15

# Terminal states
terminal.states <- list(c("x" = 4, "y" = 12))

# Rewards
rewards <- matrix(rep(-0.5, max(x)*max(y)), nrow=max(y))
rewards[ 4, 12] <-   99.5
rewards[ 6, 8] <- -100.5
rewards[ 8, 6] <- -100.5
rewards[ 8,10] <- -100.5
rewards[10, 8] <- -100.5

# Boundaries
rewards[ 6, 6] <- NA
rewards[ 6,14] <- NA
rewards[14, 6] <- NA
rewards[14,14] <- NA

# Actions are four cardinal directions of travel.
actions <- c("N", "S", "E", "W")
action.values <- list("N" = c("x" = 0, "y" = 1),
                      "S" = c("x" = 0, "y" = -1),
                      "E" = c("x" = 1, "y" = 0),
                      "W" = c("x" = -1, "y" = 0))

# Transition probability
transition <- list("N" = c("N" = 0.8, "S" = 0.0, "E" = 0.1, "W" = 0.1),
                   "S" = c("N" = 0.0, "S" = 0.8, "E" = 0.1, "W" = 0.1),
                   "E" = c("N" = 0.1, "S" = 0.1, "E" = 0.8, "W" = 0.0),
                   "W" = c("N" = 0.1, "S" = 0.1, "E" = 0.0, "W" = 0.8))

# Get result of action
act <- function(action, state) {
    for (terminal.state in terminal.states) {
        if (state['x'] == terminal.state['x'] &&
                state['y'] == terminal.state['y']) {
            return(state)
        }        
    }
    
    # Calculate new state and ensure it is within environment.
    new.state <- state
    action.value <- action.values[[action]]
    new.x <- state['x'] + action.value['x']
    new.y <- state['y'] + action.value['y']    
    new.state['x'] <- min(x[length(x)], max(x[1], new.x))
    new.state['y'] <- min(y[length(y)], max(y[1], new.y))
    
    # Checks if new state is boundary.
    if(is.na(rewards[new.state['y'], new.state['x']])) {
        return(state)
    }
    
    return(new.state)
}

# Build transition probability and reward arrays
# <S, S, A>
P <- array(0, c(max(x) * max(y),max(x) * max(y),length(actions)))
R <- array(0, c(max(x) * max(y),max(x) * max(y),length(actions)))
for (i in x) {
    for (j in y) {
        int_state = i + ((j - 1) * max(x))
        for (k in 1:length(actions)) {
            for (l in 1:length(actions)) {
                prob <- as.numeric(transition[[actions[k]]][actions[l]])
                next_state <- act(actions[l],c("x"=i,"y"=j))
                next_int_state = next_state['x'] + ((next_state['y'] - 1) * max(x))
                P[int_state,next_int_state,k] <- P[int_state,next_int_state,k] + prob
                R[int_state,next_int_state,k] <- rewards[next_state['y'],next_state['x']]
            }
        }
    }
}
# Terminal states
for (terminal.state in terminal.states) {
    int_state <- terminal.state['x'] + ((terminal.state['y'] - 1) * max(x))
    for (k in 1:length(actions)) {
        R[int_state,int_state,k] <- 100
    }
}


### Value Iteration
value_iteration_solution <- mdp_value_iterationGS(P, R, discount=0.4)

# Values at each step. These values can be different in each run
value_iteration_solution$V 

# Number of iterations
value_iteration_solution$iter

# Time Taken
value_iteration_solution$time

value_iteration_policy <- matrix(rep(0, max(x)*max(y)), nrow=max(y))
for (i in x) {
  for (j in y) {
    value_iteration_policy[j,i] <- value_iteration_solution$V[
      as.integer(i + ((j - 1) * max(x)))]
  }
}

## Create HeatMap
v <- ggplot(melt(value_iteration_policy), aes(x=Var1, y=Var2)) + ggtitle("Value Iteration: Grid World") +
  geom_tile(aes(fill = value), colour = "white") + scale_fill_gradient(low = "white", high = "steelblue")

v + geom_text(aes(Var1, Var2, label = round(value,2)), color = "black", size = 2.5) + theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  legend.position="none")


### Policy Iteration
policy_iteration_solution <- mdp_policy_iteration_modified(P, R, discount=0.4)

# View Results
names(action.values)[policy_iteration_solution$policy]

# Values at each step. These values can be different in each run
policy_iteration_solution$V

# Number of iterations
policy_iteration_solution$iter

# Time taken
policy_iteration_solution$time 

policy_iteration_policy <- matrix(rep(0, max(x)*max(y)), nrow=max(y))
for (i in x) {
  for (j in y) {
    policy_iteration_policy[j,i] <- policy_iteration_solution$V[
      as.integer(i + ((j - 1) * max(x)))]
  }
}

## Create Heatmap
p <- ggplot(melt(policy_iteration_policy), aes(x=Var1, y=Var2)) + ggtitle("Policy Iteration: Grid World") +
  geom_tile(aes(fill = value), colour = "white") + scale_fill_gradient(low = "white", high = "steelblue")

p + geom_text(aes(Var1, Var2, label = round(value,2)), color = "black", size = 2.5) + theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  legend.position="none")


### q-Learning
ptm <- proc.time()
q_learning_solution <- mdp_Q_learning(P, R, discount=0.4)
cat("Time:",(proc.time() - ptm),"\n")

# Values at each step. These values can be different in each run
q_learning_solution$V


q_learning_policy <- matrix(rep(0, max(x)*max(y)), nrow=max(y))
for (i in x) {
  for (j in y) {
    q_learning_policy[j,i] <- q_learning_solution$V[
      as.integer(i + ((j - 1) * max(x)))]
  }
}

## Graph Heatmap
q <- ggplot(melt(q_learning_policy), aes(x=Var1, y=Var2)) + ggtitle("Q-Learning: Grid World") +
  geom_tile(aes(fill = value), colour = "white") + scale_fill_gradient(low = "white", high = "steelblue")

q + geom_text(aes(Var1, Var2, label = round(value,2)), color = "black", size = 2.5) + theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  legend.position="none")
