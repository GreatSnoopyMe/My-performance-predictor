# new Predictor

## Steps:

1 choose all learning curves that may converge

1.1 set x and y for these curves, and then do the training for $\theta_i$(<div style="color:red">new methods</div>)

1.2 add regularization/...(<div style="color:red">new methods</div>)


2 after getting $\theta_i$, we need to find out 2 things

2.1 find out $w_i$ (<div style="color:red">new methods</div>)
2.1.1 Rank the fitting result by 
```
# residual sum of squares
ss_res = np.sum((y - y_fit) ** 2)

# total sum of squares
ss_tot = np.sum((y - np.mean(y)) ** 2)

# r-squared
r2 = 1 - (ss_res / ss_tot)
```

2.2 find out $x_{conv}$, where the learning curve will converge(easy and quick)

3 Now calculate $y_{predict}$

3.1 with $w_i$ and $\theta_i$, calculate $y_{combined}$

4 compare with the needed predicted position with converged $x$