Homework 5
================
DS Student
January 15, 2020

## Randomly split the mcycle data into training (75%) and validation (25%) subsets.

``` r
library('MASS')
data <- mcycle
dim(data)[1]
```

    ## [1] 133

``` r
train_data_ind <- sample(dim(data)[1], dim(data)[1]*0.75) 
train_data <- data[train_data_ind,]
test_data <- data[-train_data_ind,]
train_y <- train_data$accel
train_x <- matrix(train_data$times, length(train_data$times), 1)
test_y <- test_data$accel
test_x <- matrix(test_data$times, length(test_data$times), 1)
```

## Using the mcycle data, consider predicting the mean acceleration as a function of time. Use the Nadaraya-Watson method with the k-NN kernel function to create a series of prediction models by varying the tuning parameter over a sequence of values. (hint: the script already implements this)

``` r
Nadaraya_Watson <- function(y,x,x0,kern,...){
  a <- t(apply(x0, 1, function(x0_){
    k <- kern(x,x0_,...)
    k/sum(k)
    }))

    y_hat <- drop(a %*% y)
    attr(y_hat,"k") <- a
    return(y_hat)
}

kernel_k_nearest_neighbors <- function(x, x0, t=1) {
  ## compute distance betwen each x and x0
  z <- t(t(x) - x0)
  d <- sqrt(rowSums(z*z))

  ## initialize kernel weights to zero
  w <- rep(0, length(d))
  
  ## set weight to 1 for k nearest neighbors
  w[order(d)[1:t]] <- 1
  
  return(w)
}
k1 <- seq(1,20,1)
for (i in k1){
  y_hat <- Nadaraya_Watson(y = train_y,x = train_x,x0 = test_x, kern = kernel_k_nearest_neighbors, t = i)
}
```

## With the squared-error loss function, compute and plot the training error, AIC, BIC, and validation error (using the validation data) as functions of the tuning parameter.

``` r
effective_df <- function(y, x, kern, ...) {
  y_hat <- Nadaraya_Watson(y, x, x,
    kern=kern, ...)
  sum(diag(attr(y_hat, 'k')))
}


loss_squared_error <- function(y, yhat)
  (y - yhat)^2

error <- function(y, yhat, loss=loss_squared_error)
  mean(loss(y, yhat))

aic <- function(y, yhat, d)
  error(y, yhat) + 2/length(y)*d

bic <- function(y, yhat, d)
  error(y, yhat) + log(length(y))/length(y)*d

AIC <- rep(0,1)
BIC <- rep(0,1)
training_error <- rep(0,1)
testing_error <- rep(0,1)
s = 1

for (i in (1:length(k1))){
  edf <- effective_df(train_y, train_x, kernel_k_nearest_neighbors, t=k1[i])
  y_hat_train <- Nadaraya_Watson(y = train_y,x = train_x,x0 = train_x, kern = kernel_k_nearest_neighbors, t = k1[i])
  y_hat_test <- Nadaraya_Watson(y = train_y,x = train_x,x0 = test_x, kern = kernel_k_nearest_neighbors, t = k1[i])
  AIC[i] <- aic(train_y, y_hat_train,edf)
  BIC[i] <- bic(train_y, y_hat_train,edf)
  training_error[i] <- error(train_y, y_hat_train)
  testing_error[i] <- error(test_y, y_hat_test)
}
print(training_error)
```

    ##  [1] 206.3732 258.0959 315.8020 349.2357 376.5985 417.3099 415.7433 430.0148
    ##  [9] 446.1634 482.6777 496.6120 536.5947 554.4745 561.0305 572.9146 597.5016
    ## [17] 605.4055 634.0140 632.9927 660.0885

``` r
print(testing_error)
```

    ##  [1] 1157.0232 1025.0303  987.1629  717.0585  627.0149  586.2903  578.4574
    ##  [8]  572.7066  567.7348  531.4433  535.4938  484.8347  460.8144  450.3798
    ## [15]  464.4422  493.0336  519.5226  550.3141  586.2618  587.6376

``` r
print(AIC)
```

    ##  [1] 207.9086 259.0252 316.4417 349.7306 376.9985 417.6432 416.0290 430.2648
    ##  [9] 446.3857 482.8777 496.7938 536.7614 554.6284 561.1733 573.0479 597.6266
    ## [17] 605.5231 634.1251 633.0980 660.1885

``` r
print(BIC)
```

    ##  [1] 209.9008 260.2310 317.2718 350.3729 377.5175 418.0757 416.3997 430.5892
    ##  [9] 446.6740 483.1373 497.0297 536.9777 554.8280 561.3587 573.2209 597.7888
    ## [17] 605.6758 634.2693 633.2346 660.3183

``` r
plot(k1,training_error,type = "l", col = "green",ylim = c(180,1300),xlab = "k",ylab = "error")
lines(k1,AIC,col = "pink",type = "l")
lines(k1,BIC,col = "blue",type = "l")
lines(k1,testing_error,col = "red",type = "l")
legend("topright", c("training error", "AIC", "BIC", "Validation error"), col = c("green","pink","blue","red"))
```

![](HW4_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

## For each value of the tuning parameter, Perform 5-fold cross-validation using the combined training and validation data. This results in 5 estimates of test error per tuning parameter value.

``` r
f <- 5
k1 <- seq(1,20,1)
folds <- sample(rep(1:5,length(data)))
err <- rep(0,1)
for (i in (1:length(k1))){
  e <- rep(0,5)
  for (j in (1:5)){
    train_y <- train_data$accel[folds!= j]
    train_x <- matrix(data$times[folds!= j], length(train_data$times[folds!= j]), 1)
    test_y <- test_data$accel
    test_x <- matrix(test_data$times[folds== j], length(test_data$times[folds== j]), 1)
    # train model
    y_hat <- Nadaraya_Watson(y = train_y,x = train_x,x0 = test_x, kern = kernel_k_nearest_neighbors, t = k1[i])
    # error of every validation
    
    e[j] <- error(test_y, y_hat)
  }
  print(e)
  err[i] <- mean(e)
}
```

    ## [1] 4408.867 4274.677 4555.041 4361.560 4462.434
    ## [1] 3252.128 3579.964 2744.569 3073.637 3050.424
    ## [1] 3277.795 3483.374 2550.052 3616.570 2884.295
    ## [1] 3000.104 2992.686 2620.350 3163.423 2749.378
    ## [1] 3414.756 2790.960 2663.963 3110.236 3040.683
    ## [1] 3172.024 2726.324 2719.931 2872.114 2744.108
    ## [1] 3018.131 2532.825 2508.371 2622.385 2580.959
    ## [1] 2786.435 2418.171 2496.358 2561.794 2461.484
    ## [1] 2676.259 2509.430 2516.608 2535.887 2455.330
    ## [1] 2537.131 2373.706 2442.023 2519.731 2395.636
    ## [1] 2523.967 2339.953 2370.482 2458.372 2396.886
    ## [1] 2476.556 2303.161 2375.573 2384.730 2392.585
    ## [1] 2494.542 2337.553 2400.879 2368.353 2383.467
    ## [1] 2485.928 2394.753 2406.976 2364.362 2436.435
    ## [1] 2487.767 2398.276 2408.786 2358.694 2426.302
    ## [1] 2416.038 2409.569 2446.008 2340.728 2395.455
    ## [1] 2410.636 2460.657 2426.817 2343.871 2394.016
    ## [1] 2412.297 2432.104 2450.240 2354.036 2392.276
    ## [1] 2429.449 2442.413 2416.684 2347.750 2400.306
    ## [1] 2411.619 2444.769 2381.521 2331.496 2406.315

## Plot the CV-estimated test error (average of the five estimates from each fold) as a function of the tuning parameter. Add vertical line segments to the figure (using the segments function in R) that represent one “standard error” of the CV-estimated test error (standard deviation of the five estimates from each fold).

``` r
sd <- sd(err)
plot(k1,err,type = "l",col = "red", ylim = c(500,6000))
abline(h = sd)
legend("topright",c("error", "sd"), col = c("red", "black"),lty = 1 )
```

![](HW4_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

## Interpret the resulting figures and select a suitable value for the tuning parameter.

From the plot we can see that when k = 10, the test and train plot is
the smallest. When k \>10 and k increases, test and train error also
increase. Waht’s more, AIC, BIC are similar with the train error.
Therefore, suitable k value is about 10.
