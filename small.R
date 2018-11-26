# Mindy White
# CS 7641: Machine Learning
# Markov Decision Processes
# Small States: Golf Problem
#
# Initialize Environment
install.packages("MDPtoolbox")
install.packages("ggplot2")
install.packages("reshape2")
require("MDPtoolbox");require(ggplot2);require(reshape2)


#
# Define world as a 5x10 grid
golf_x <- 1:10
golf_y <- 1:5

# Terminal states
terminal.states <- list(c("x" = 9, "y" = 5))

# Rewards
rewards <- matrix(rep(-1, max(golf_x)*max(golf_y)), nrow=max(golf_y))
rewards[5,   9] <-  4
rewards[5, 1:4] <- -2
rewards[4, 6:10] <- -2
rewards[3,  7:9] <- -2
rewards[2,2:4] <- -2
rewards[1,8:10] <- -2

# Actions are four cardinal directions of travel.
actions <- c("N_P",  "N_M",  "N_L",  "N_P",
             "NE_P", "NE_M", "NE_L", "NE_P",
             "E_P",  "E_M",  "E_L",  "E_P",
             "SE_P", "SE_M", "SE_L", "SE_P",
             "S_P",  "S_M",  "S_L",  "S_P",
             "SW_P", "SW_M", "SW_L", "SW_P",
             "W_P",  "W_M",  "W_L",  "W_P",
             "NW_P", "NW_M", "NW_L", "NW_P")
action.values <- list("N_P" = c("x" = 0, "y" = 1),
                      "N_L" = c("x" = 0, "y" = 2),
                      "N_M" = c("x" = 0, "y" = 5),
                      "N_D" = c("x" = 0, "y" = 10),
                      "NE_P" = c("x" = ceiling(1/sqrt(2)),  "y" = ceiling(1/sqrt(2))),
                      "NE_L" = c("x" = ceiling(2/sqrt(2)),  "y" = ceiling(2/sqrt(2))),
                      "NE_M" = c("x" = ceiling(5/sqrt(2)),  "y" = ceiling(5/sqrt(2))),
                      "NE_D" = c("x" = ceiling(10/sqrt(2)), "y" = ceiling(10/sqrt(2))),
                      "E_P" = c("x" = 1,  "y" = 0),
                      "E_L" = c("x" = 2,  "y" = 0),
                      "E_M" = c("x" = 5,  "y" = 0),
                      "E_D" = c("x" = 10, "y" = 0),
                      "SE_P" = c("x" = ceiling(1/sqrt(2)),  "y" = -ceiling(1/sqrt(2))),
                      "SE_L" = c("x" = ceiling(2/sqrt(2)),  "y" = -ceiling(2/sqrt(2))),
                      "SE_M" = c("x" = ceiling(5/sqrt(2)),  "y" = -ceiling(5/sqrt(2))),
                      "SE_D" = c("x" = ceiling(10/sqrt(2)), "y" = -ceiling(10/sqrt(2))),
                      "S_P" = c("x" = 0, "y" = -1),
                      "S_M" = c("x" = 0, "y" = -2),
                      "S_L" = c("x" = 0, "y" = -5),
                      "S_P" = c("x" = 0, "y" = -10),
                      "SW_P" = c("x" = -ceiling(1/sqrt(2)),  "y" = -ceiling(1/sqrt(2))),
                      "SW_L" = c("x" = -ceiling(2/sqrt(2)),  "y" = -ceiling(2/sqrt(2))),
                      "SW_M" = c("x" = -ceiling(5/sqrt(2)),  "y" = -ceiling(5/sqrt(2))),
                      "SW_D" = c("x" = -ceiling(10/sqrt(2)), "y" = -ceiling(10/sqrt(2))),
                      "W_P" = c("x" = -1, "y" = 0),
                      "W_M" = c("x" = -2, "y" = 0),
                      "W_L" = c("x" = -5, "y" = 0),
                      "W_P" = c("x" = -10, "y" = 0),
                      "NW_P" = c("x" = -ceiling(1/sqrt(2)),  "y" = ceiling(1/sqrt(2))),
                      "NW_L" = c("x" = -ceiling(2/sqrt(2)),  "y" = ceiling(2/sqrt(2))),
                      "NW_M" = c("x" = -ceiling(5/sqrt(2)),  "y" = ceiling(5/sqrt(2))),
                      "NW_D" = c("x" = -ceiling(10/sqrt(2)), "y" = ceiling(10/sqrt(2))))

# Build transition probability and reward arrays
# <S, S, A>
P <- array(0, c(max(golf_x) * max(golf_y),max(golf_x) * max(golf_y),length(actions)))
R <- array(0, c(max(golf_x) * max(golf_y),max(golf_x) * max(golf_y),length(actions)))
for (i in golf_x) {
  for (j in golf_y) {
    start.state <- c("x"=i,"y"=j)
    start.state.int <- i + ((j - 1) * max(golf_x))
    
    is.terminal <- 0
    for (terminal.state in terminal.states) {
      if (start.state['x'] == terminal.state['x'] &&
          start.state['y'] == terminal.state['y']) {
        P[start.state.int,start.state.int,1:length(actions)] <- 1
        R[start.state.int,start.state.int,1:length(actions)] <- 4
        is.terminal <- 1
        break
      }        
    }
    if (is.terminal == 1) break
    
    for (k in 1:length(actions)) {
      action.value <- action.values[[actions[k]]]
      new.x <- start.state['x'] + action.value['x']
      new.y <- start.state['y'] + action.value['y']
      
      for (n in new.x-1:new.x+1) {
        for (m in new.y-1:new.y+1) {
          if (n == new.x && m == new.y) prob <- 0.6
          else prob <- 0.05
          
          next.state <- start.state
          next.state['x'] <- min(golf_x[length(golf_x)], max(golf_x[1], n))
          next.state['y'] <- min(golf_y[length(golf_y)], max(golf_y[1], m))
          next.state.int = next.state['x'] + ((next.state['y'] - 1) * max(golf_x))
          
          P[start.state.int,next.state.int,k] <- P[start.state.int,next.state.int,k] + prob
          R[start.state.int,next.state.int,k] <- rewards[next.state['y'],next.state['x']]
        }
      }
      
    }
  }
}


### Value Iteration
value_iteration_solution <- mdp_value_iterationGS(P, R, discount=0.2)

## View Results
value_iteration_solution$policy
names(action.values)[value_iteration_solution$policy]

# Values at each step. These values can be different in each run
value_iteration_solution$V 

# Number of iterations
value_iteration_solution$iter

# Time Taken
value_iteration_solution$time

value_iteration_policy <- matrix(rep(0, max(golf_x)*max(golf_y)), nrow=max(golf_y))
for (i in golf_x) {
  for (j in golf_y) {
    value_iteration_policy[j,i] <- value_iteration_solution$V[
      as.integer(i + ((j - 1) * max(golf_x)))]
  }
}

## Create HeatMap
v <- ggplot(melt(value_iteration_policy), aes(x=Var1, y=Var2)) + ggtitle("Value Iteration: Golf Problem") +
  geom_tile(aes(fill = value), colour = "white") + scale_fill_gradient(low = "white", high = "green")

v + geom_text(aes(Var1, Var2, label = round(value,2)), color = "black", size = 4) + theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  legend.position="none")


### Policy Iteration
policy_iteration_solution <- mdp_policy_iteration_modified(P, R, discount=0.2)

# View Results
names(action.values)[policy_iteration_solution$policy]

# Values at each step. These values can be different in each run
policy_iteration_solution$V

# Number of iterations
policy_iteration_solution$iter

# Time taken
policy_iteration_solution$time 

policy_iteration_policy <- matrix(rep(0, max(golf_x)*max(golf_y)), nrow=max(golf_y))
for (i in golf_x) {
  for (j in golf_y) {
    policy_iteration_policy[j,i] <- policy_iteration_solution$V[
      as.integer(i + ((j - 1) * max(golf_x)))]
  }
}

## Create Heatmap
p <- ggplot(melt(policy_iteration_policy), aes(x=Var1, y=Var2)) + ggtitle("Policy Iteration: Golf Problem") +
  geom_tile(aes(fill = value), colour = "white") + scale_fill_gradient(low = "white", high = "green")

p + geom_text(aes(Var1, Var2, label = round(value,2)), color = "black", size = 4) + theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  legend.position="none")


### q-Learning
ptm <- proc.time()
q_learning_solution <- mdp_Q_learning(P, R, discount=0.2)
cat("Time:",(proc.time() - ptm),"\n")

# Values at each step. These values can be different in each run
q_learning_solution$V


q_learning_policy <- matrix(rep(0, max(golf_x)*max(golf_y)), nrow=max(golf_y))
for (i in golf_x) {
  for (j in golf_y) {
    q_learning_policy[j,i] <- q_learning_solution$V[
      as.integer(i + ((j - 1) * max(golf_x)))]
  }
}

## Graph Heatmap
q <- ggplot(melt(q_learning_policy), aes(x=Var1, y=Var2)) + ggtitle("Q-Learning: Golf Problem") +
  geom_tile(aes(fill = value), colour = "white") + scale_fill_gradient(low = "white", high = "green")

q + geom_text(aes(Var1, Var2, label = round(value,2)), color = "black", size = 4) + theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  legend.position="none")
