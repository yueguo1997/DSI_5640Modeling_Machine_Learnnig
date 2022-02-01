hw2
================
Yue Guo

# Homework02

## Background

``` r
library("qrnn")
## load prostate data
prostate <- 
  read.table(url(
    'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'))

## subset to training examples
prostate_train <- subset(prostate, train==TRUE)

## plot lcavol vs lpsa
plot_psa_data <- function(dat=prostate_train) {
  plot(dat$lpsa, dat$lcavol,
       xlab="log Prostate Screening Antigen (psa)",
       ylab="log Cancer Volume (lcavol)",
       pch = 20)
}
plot_psa_data()

############################
## regular linear regression
############################

## L2 loss function
L2_loss <- function(y, yhat)
  (y-yhat)^2

## fit simple linear model using numerical optimization
predict_lin <- function(x, beta){
  beta[1] + beta[2]*x
}
fit_lin <- function(y, x, loss=L2_loss, beta_init = c(-0.51, 0.75), predict_fun = predict_lin) {
  err <- function(beta)
    mean(loss(y,  predict_lin(x,beta)))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

## make predictions from linear model
 

## fit linear model
lin_beta <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L2_loss)

## compute predictions for a grid of inputs
x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
lin_pred <- predict_lin(x=x_grid, beta=lin_beta$par)

## plot data
plot_psa_data()

## plot predictions
lines(x=x_grid, y=lin_pred, col='darkgreen', lwd=2)

## do the same thing with 'lm'
lin_fit_lm <- lm(lcavol ~ lpsa, data=prostate_train)

## make predictins using 'lm' object
lin_pred_lm <- predict(lin_fit_lm, data.frame(lpsa=x_grid))

## plot predictions from 'lm'
lines(x=x_grid, y=lin_pred_lm, col='pink', lty=2, lwd=2)
```

![](Untitled_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
##################################
## try modifying the loss function
##################################

## custom loss function
custom_loss <- function(y, yhat){
  (y-yhat)^2 + abs(y-yhat)
  
}
  

## plot custom loss function
err_grd <- seq(-1,1,length.out=200)
plot(err_grd, custom_loss(err_grd,0), type='l',
     xlab='y-yhat', ylab='custom loss')
```

![](Untitled_files/figure-gfm/unnamed-chunk-1-2.png)<!-- -->

``` r
## fit linear model with custom loss
lin_beta_custom <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=custom_loss)

lin_pred_custom <- predict_lin(x=x_grid, beta=lin_beta_custom$par)

## plot data
plot_psa_data()

## plot predictions from L2 loss
lines(x=x_grid, y=lin_pred, col='darkgreen', lwd=2)

## plot predictions from custom loss
lines(x=x_grid, y=lin_pred_custom, col='pink', lwd=2, lty=2)
```

![](Untitled_files/figure-gfm/unnamed-chunk-1-3.png)<!-- -->

## Question 1

Question: Write functions that implement the L1 loss and tilted absolute
loss functions.

``` r
absolute_loss_function <- function(y,yhat,tau){
  tilted.abs(y-yhat,tau)
}
```

## Qustion 2

Question: Create a figure that shows lpsa (x-axis) versus lcavol
(y-axis). Add and label (using the ‘legend’ function) the linear model
predictors associated with L2 loss, L1 loss, and tilted absolute value
loss for tau = 0.25 and 0.75.

``` r
fit_lin_new <- function(y, x, loss=L2_loss, beta_init = c(-0.51, 0.75), predict_fun = predict_lin,tau = 0.5) {
  err <- function(beta)
    mean(loss(y,  predict_lin(x,beta),tau))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

abs_beta_25 <- fit_lin_new(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    beta_init = c(-0.51, 0.75),
                    loss= absolute_loss_function,
                    tau = 0.25)
abs_beta_75 <- fit_lin_new(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    beta_init = c(-0.51, 0.75),
                    loss= absolute_loss_function,
                    tau = 0.75)

abs_pred_25 <- predict_lin(x=x_grid, beta=abs_beta_25$par)
abs_pred_75 <- predict_lin(x=x_grid, beta=abs_beta_75$par)
plot_psa_data()
title = (main = "Linear prediction")
lines(x=x_grid, y=lin_pred, col='darkgreen', lwd=2)
lines(x=x_grid, y=abs_pred_25, col='pink', lwd=2)
lines(x=x_grid, y=abs_pred_75, col='blue', lwd=2)
legend("bottomright", legend = c("L2_loss", "L1_loss_tau=0.25", "L1_loss_tau=0.75"),col = c("darkgreen","pink","blue"), lty  = 1)
```

![](Untitled_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

## Qustion 3

Question: Write functions to fit and predict from a simple nonlinear
model with three parameters defined by ‘beta\[1\] +
beta\[2\]*exp(-beta\[3\]*x)’. Hint: make copies of ‘fit_lin’ and
‘predict_lin’ and modify them to fit the nonlinear model. Use c(-1.0,
0.0, -0.3) as ‘beta_init’.

``` r
nonlinear_pred <- function(x,beta){
  beta[1] + beta[2]*exp(-beta[3]*x)
}

fit_nonlin <- function(y, x, loss=L2_loss, beta_init = c(-1.0, 0.0, -0.3), predict_fun = nonlinear_pred){
    err <- function(beta)
    mean(loss(y,  predict_lin(x,beta)))
    beta <- optim(par = beta_init, fn = err)
    return(beta)
}
fit_nonlin_abs <- function(y, x, loss=L2_loss, beta_init = c(-1.0, 0.0, -0.3), predict_fun = nonlinear_pred,tau){
    err <- function(beta)
    mean(loss(y,  predict_lin(x,beta),tau))
    beta <- optim(par = beta_init, fn = err)
    return(beta)
    }
```

## Qustion 4

Question: Create a figure that shows lpsa (x-axis) versus lcavol
(y-axis). Add and label (using the ‘legend’ function) the nonlinear
model predictors associated with L2 loss, L1 loss, and tilted absolute
value loss for tau = 0.25 and 0.75.

``` r
L1_non_pred_beta_25 <- fit_nonlin_abs(y=prostate_train$lcavol,
                               x=prostate_train$lpsa, 
                               loss=absolute_loss_function, 
                               beta_init = c(-1.0, 0.0, -0.3),
                               predict_fun = nonlinear_pred,tau = 0.25)
L1_non_pred_beta_75 <- fit_nonlin_abs(y=prostate_train$lcavol,
                               x=prostate_train$lpsa, 
                               loss=absolute_loss_function, 
                               beta_init = c(-1.0, 0.0, -0.3),
                               predict_fun = nonlinear_pred,tau = 0.75)
                                           
L2_non_pred_beta <- fit_nonlin(y=prostate_train$lcavol,
                               x=prostate_train$lpsa,
                               loss=L2_loss, 
                               beta_init = c(-1.0, 0.0, -0.3), 
                               predict_fun = nonlinear_pred)

L1_non_pred_25 <- nonlinear_pred(x=x_grid, beta=L1_non_pred_beta_25$par)
L1_non_pred_75 <- nonlinear_pred(x=x_grid, beta=L1_non_pred_beta_75$par)
L2_non_pred <- nonlinear_pred(x=x_grid, beta=L2_non_pred_beta$par)
plot_psa_data()
title(main = "Nonlinear prediction")
## plot predictions from L2 loss
lines(x=x_grid, y=L1_non_pred_25, col='darkgreen', lwd=2)
lines(x=x_grid, y=L1_non_pred_75, col='blue', lwd=2)
## plot predictions from custom loss
lines(x=x_grid, y=L2_non_pred, col='pink', lwd=2)
legend("bottomright", "(x,y)", legend = c("L1_Loss_tau = 0.25","L1_Loss_tau = 0.75", "L2_loss"), col = c("darkgreen","blue", "pink"),lwd = 1)
```

![](Untitled_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->
