TITLE EXAMPLE


## Part 1.1: Non-Convex Function with Single Variable Optimization

First, we generate functions representing Vanilla gradient descent and Adam algorithms for comparison and performance monitoring. Vanilla gradient descent algorithm of choice is the same as the one presented in class. We set default parameters for learning rate = 0.005 and max iterations = 200.

```{r, echo=FALSE, warning=FALSE, message=FALSE}

gradient_desc <- function(f, x_init, learning_rate_gd=0.005, max_iters_gd=200){
    
    # Parameters
    iters_gd <- 0
    
    # Starting conditions
    x <- rep(0,max_iters_gd)
    x[1] <- x_init
    f_der <- Deriv(f)
    
    for (i in 2:max_iters_gd){
        
        # Calculate gradient
        grad <- f_der(x[i-1])
        x[i] <- x[i-1]-learning_rate_gd*grad
        iters_gd <- iters_gd + 1
        
        # Stopping criteria
        if(sum(abs(x[i]-x[i-1])) < 0.001){
            x[i:max_iters_gd] <- x[i]
            break
        }
    }
    return(x)
}

```


```{r, echo=FALSE, warning=FALSE, message=FALSE}

adam_func <- function(f, x_init, learning_rate_ad=0.005, max_iters_ad=200){
    
    # Parameters
    iters_ad <- 0
    beta1 <- 0.9
    beta2 <- 0.999
    eps <- 1e-8
    
    # Starting conditions
    x <- rep(0,max_iters_ad)
    x[1] <- x_init
    m <- 0
    v <- 0
    f_der <- Deriv(f)
    
    for (i in 2:max_iters_ad){
        grad <- f_der(x[i-1])
        m <- beta1*m + (1-beta1)*grad
        v <- beta2*v + (1-beta2)*grad^2
        m_hat <- m/(1-beta1^i)
        v_hat <- v/(1-beta2^i)
        x[i] <- x[i-1] - learning_rate_ad*m_hat/(sqrt(v_hat) + eps)
    }
    return(x)
}

```

Adam is particularly advantageous for minimizing certain non-convex functions. We generate a few example functions and provide examples for comparing the two. We then generate example functions for comparing the two algorithms. 
