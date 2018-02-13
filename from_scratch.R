

##################################################################################
################################# NEURAL NETWORK #################################
##################################################################################

# Data shape: 
# Observations are in rows while features are in columns.

# Target shape:
# Observations are in columns 
#(Only one row for regression and one for each class for classification) 

Neural_network <- function(data, target, hidden_layer, activation, iterations, alpha, lambda = 10^-3)
{
  #### PARAMETERS ####
  if (is.vector(target) == TRUE)
  {
    dim <- 1
  } else {
    dim <- length(target[,1])
  }
  
  m <- nrow(data)
  n <- ncol(data)
  K <- c(n, hidden_layer, dim)
  layer <- length(K)
  
  #### VARIABLES INITIALISATION ####
  theta <- list()
  for (i in 1:(layer-1))
  {
    theta[[i]] <- matrix(runif(K[i+1]*(K[i]+1),-0.1,0.1), nrow = K[i+1], ncol = K[i] + 1)
  }
  
  A <- list()
  delta_min <- list()
  delta_maj <- list()
  gradient <- list()
  Cost <- vector()
  
  #### COST FUNCTION ####
  
  # CLASSIFICATION vs REGRESSION
  if (activation == "logistic")
  {
    cost_function <- function(theta, output, target, lambda)
    {
      nb_theta <- length(theta)
      m <- ncol(target)
      
      J <- 0
      for (k in 1:m)
      {
        J <- J + (target[,k] %*% t(t(log(output)[,k])) + ((1 - target[,k]) %*% t(t(log(1 - output)[,k]))))
      }
      J <- -(1/m) * J
      
      # Regularization
      penalized <- 0
      for (i in 1:(nb_theta))
      {
        penalized <- penalized + sum(theta[[i]]^2)
      }
      J <- J + (lambda/(2*m)) * penalized
      
      return(J)
    }
    
    logistic <- function(z)
    {
      return(1/(1 + exp(-z)))
    }
  } 
  
  if (activation == "lineaire")
  {  
    cost_function <- function(theta, output, target, lambda)
    {
      nb_theta <- length(theta)
      m <- length(target)
      
      J <- (1/(2*m)) * sum((output - target)^2)
      
      # Regularization
      penalized <- 0
      for (i in 1:(nb_theta))
      {
        penalized <- penalized + sum(theta[[i]]^2)
      }
      J <- J + (lambda/(2*m)) * penalized  
        
      return(J)
    }
  }
  
  #### ALGORITHM ####
  
  ### LOGISTIC
  if (activation == "logistic")
  {
    for (iter in 1:iterations)     
    {
      cat("Processing step", iter)
      
      # Forward propagation
      A[[1]] <- t(data)
      for (i in 1:(layer-1))
      {
        A[[i]] <- rbind(1,A[[i]])
        A[[i+1]] <- logistic(theta[[i]] %*% A[[i]]) 
      }
      
      output <- A[[layer]]
      Cost[iter] <- cost_function(theta, output, target, lambda)
      cat("Cost : ", round(Cost[iter],4)," \n")
      
      # Backpropagation
      delta_min[[layer]] <- A[[layer]] - target
      for (i in (layer-1):2)
      {
        delta_min[[i]] <- t(theta[[i]]) %*% delta_min[[i+1]] * A[[i]] * (1 - A[[i]])
        delta_min[[i]] <- delta_min[[i]][-1,]              # Remove delta_min 0
      }
      
      for (i in 1:(layer-1))
      {
        delta_maj[[i]] <- delta_min[[i+1]] %*% t(A[[i]])
        gradient[[i]] <- (1/m) * (delta_maj[[i]] + lambda * theta[[i]])
      }
      
      # Gradient descent
      for (i in 1:(layer-1))
      {
        theta[[i]] <- theta[[i]] - alpha * gradient[[i]]
      }
    }
  } 
  
  if (activation == "lineaire")
  {
    for (iter in 1:iterations)      
    {
      cat("Processing step", iter)
      
      # Forward propagation
      A[[1]] <- t(data)
      for (i in 1:(layer-1))
      {
        A[[i]] <- rbind(1,A[[i]])
        A[[i+1]] <- theta[[i]] %*% A[[i]]
      }
      
      output <- A[[layer]]
      Cost[iter] <- cost_function(theta, output, target, lambda)
      cat("Cost : ", round(Cost[iter],4)," \n")
      
      # Backpropagation
      delta_min[[layer]] <- A[[layer]] - target
      for (i in (layer-1):2)
      {
        delta_min[[i]] <- t(theta[[i]]) %*% delta_min[[i+1]]
        delta_min[[i]] <- delta_min[[i]][-1,]              # Remove delta_min 0
      }
      
      for (i in 1:(layer-1))
      {
        delta_maj[[i]] <- delta_min[[i+1]] %*% t(A[[i]])
        gradient[[i]] <- (1/m) * (delta_maj[[i]] + lambda * theta[[i]])
      }
      
      # Gradient descent
      for (i in 1:(layer-1))
      {
        theta[[i]] <- theta[[i]] - alpha * gradient[[i]]
      }
    }
  }
  
  theta <<- theta
  Cost <<- Cost
  
}




