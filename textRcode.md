
## R Code Used in [Time Series Analysis and Its Applications, 5th Edition](https://github.com/nickpoison/tsa5)   - tsa5

<img align="left" src="https://github.com/nickpoison/astsa/blob/master/fun_with_astsa/figs/tsa4.jpg" alt="&nbsp; tsa5 &nbsp;"  height="200"/>

<br/>


 &#x2728; See the [NEWS](https://github.com/nickpoison/astsa/blob/master/NEWS.md) for further details about the state of the `astsa` package and the changelog.  

 &#10024; An intro to `astsa` capabilities can be found at  [FUN WITH ASTSA](https://github.com/nickpoison/astsa/blob/master/fun_with_astsa/fun_with_astsa.md)  

 &#10024; Here is [A Road Map](https://nickpoison.github.io/) if you want a broad view of what is available.  

 &#10024; A brief [R tutorial](https://dsstoffer.github.io/Rtoot)  

 &#10024; Pages for the old [4th Edition](https://github.com/nickpoison/tsa4)  

 <br/>

&#9940; &#9940;  __WARNING:__   If loaded, the package `dplyr` may (and most likely will) corrupt the base scripts  `filter` 
and  `lag`  that we use often. In this case, to avoid problems, either detach the problem package

```r
detach(package:dplyr)

```

or issue the commands 

```r
filter = stats::filter
lag = stats::lag

```
before analyzing time series data.  &#128534; If you are wondering how it is possible to corrupt a base package, you are not alone. 


<br/>

### Table of Contents

  * [Chapter 1 - Characteristics of Time Series](#chapter-1)
  * [Chapter 2 - Time Series Regression and Exploratory Data Analysis](#chapter-2)
  * [Chapter 3 - ARIMA Models](#chapter-3)
  * [Chapter 4 - Spectral Analysis and Filtering](#chapter-4)
  * [Chapter 5 - Additional Time Domain Topics](#chapter-5)
  * [Chapter 6 - State Space Models](#chapter-6)
  * [Chapter 7 - Statistical Methods in the Frequency Domain](#chapter-7)


---
---

>  __Note__ when you are in a code block below, you can copy the contents of the block by moving your mouse to the upper right corner and clicking on the copy icon ( &#128203; ).


---
---

<br/>

## Chapter 1


  Example 1.1 

```r
par(mfrow=2:1)
tsplot(jj, col=4, ylab="USD", type="o", main="Johnson & Johnson Quarterly Earning per Share")
tsplot(jj, col=4, ylab="USD", type="o", log="y")

```

<br/> Example 1.2  

```r
tsplot(cbind(gtemp_land, gtemp_ocean), spaghetti=TRUE, pch=c(20,18), type="o", col=astsa.col(c(4,2),.5), ylab="\u00B0C", main="Global Annual Mean Temperature Change")
legend("topleft", legend=c("Land Surface","Sea Surface"), lty=1, pch=c(20,18), col=c(4,2), bg="white")

```

<br/> Example 1.3  

```r
tsplot(speech, col=4)
arrows(658, 3850, 766, 3850, code=3, angle=90, length=.05, col=6)
text(712, 4100, "pitch period", cex=.75) 

```

<br/> Example 1.4  

```r
library(xts)
djia_return = diff(log(djia$Close))
par(mfrow=2:1)
plot(djia$Close, col=4, main="DJIA Close")
plot(djia_return, col=4, main="DJIA Returns")

```

<br/> Example 1.5  

```r
par(mfrow = c(2,1))  # set up the graphics
tsplot(soi, col=4, ylab="", main="Southern Oscillation Index")
text(1970,  .91, "COOL", col=5, font=4)
text(1970, -.91, "WARM", col=6, font=4)
tsplot(rec, col=4, ylab="", main="Recruitment") 

```

<br/> Example 1.6

```r
tsplot(cbind(Hare, Lynx), col=c(2,4), type="o", pch=c(0,2), ylab="Number", spaghetti=TRUE)
mtext("(\u00D7 1000)", side=2, adj=1, line=1.5, cex=.8)
legend("topright", col=c(2,4), lty=1, pch=c(0,2), legend=c("Hare", "Lynx"), bty="n")

```

<br/> Example 1.7

```r
par(mfrow=c(3,1))
x = ts(fmri1[,4:9], start=0, freq=32)  # data
u = ts(rep(c(rep(.6,16), rep(-.6,16)), 4), start=0, freq=32) # stimulus signal
names = c("Cortex","Thalamus","Cerebellum")
for (i in 1:3){
 j = 2*i-1
 tsplot(x[,j:(j+1)], ylab="BOLD", xlab="", main=names[i], col=5:6, ylim=c(-.6,.6), lwd=2, xaxt="n", spaghetti=TRUE)
 axis(seq(0,256,64), side=1, at=0:4)
 lines(u, type="s", col=gray(.3))
}
mtext("seconds", side=1, line=1.75, cex=.9)

```

<br/> Example 1.8

```r
tsplot(cbind(EQ5,EXP6), ylab=c("Earthquake", "Explosion"), col=4)

```


<br/> Example 1.10

```r
w = rnorm(250,0,1)                  # 250 N(0,1) variates
v = filter(w, sides=2, rep(1/3,3))  # moving average 
par(mfrow=c(2,1))
tsplot(w, main="white noise", col=4, gg=TRUE)
tsplot(v, ylim=c(-3,3), main="moving average", col=4, gg=TRUE)

```

<br/> Example 1.11

```r
w = rnorm(300,0,1)  # 250 +50 extra to avoid startup problems
x = filter(w, filter=c(1.5,-.75), method="recursive")[-(1:50)]
tsplot(x, col=4, main="autoregression", gg=TRUE)

```

<br/> Example 1.12

```r
set.seed(154)  # so you can reproduce the results
w  = rnorm(200);  x = cumsum(w)  # two commands in one line
wd = w +.2;      xd = cumsum(wd)
tsplot(xd, ylim=c(-5,55), main="random walk", ylab="", col=4, gg=TRUE)
lines(x, col=6);  clip(0, 200, 0, 50)
abline(h=0, a=0, b=.2, col=8, lty=5)

```

<br/> Example 1.13

```r
cs = 2*cos(2*pi*(1:500 + 15)/50)  
w = rnorm(500,0,1)
par(mfrow=c(3,1))
tsplot(cs, ylab="", main=bquote(2*cos(2*pi*t/50+.6*pi)), col=4, gg=TRUE)
tsplot(cs+w, ylab="", main=bquote(2*cos(2*pi*t/50+.6*pi) + N(0,1)), col=4, gg=TRUE)
tsplot(cs+5*w, ylab="", main=bquote(2*cos(2*pi*t/50+.6*pi) + N(0,5^2)), col=4, gg=TRUE)

```

<br/> Example 1.25

```r
set.seed(2)
x = rnorm(100)
y = lag(x, -5) + rnorm(100)
ccf2(y, x, lwd=2, col=4, type='covariance', gg=TRUE)
text( 10, 1.1, 'x leads')
text(-10, 1.1, 'y leads')

```

<br/> Marginal normals that are not bivariate normal

```r
x = rnorm(1000)
z = rnorm(1000)
y = ifelse(x*z > 0, z, -z)
scatter.hist(x, y, hist.col=5, pt.col=6)

```

<br/> Example 1.26

```r
(r = format(acf1(soi, 6, plot=FALSE), digits=2)) # first 6 sample acf values
par(mfrow=c(1,2))
tsplot(lag(soi,-1), soi, col=4, type="p", xlab="lag(soi,-1)")
 legend("topleft", legend=bquote(hat(rho)(1) == .(r[1])), bty="n", adj=.2)
tsplot(lag(soi,-6), soi, col=4, type="p", xlab="lag(soi,-6)")
 legend("topleft", legend=bquote(hat(rho)(6) == .(r[6])), bty="n", adj=.2)

```

<br/> Property 1.2 demonstration

```r
x = replicate(1000, acf1(rnorm(100), plot=FALSE))
round(c(mean(x), sd(x)), 3)
qqnorm(x); qqline(x)  # to check normality (not shown)

```


<br/> Example 1.27

```r
set.seed(101011)
x    = sample(c(-2,2), 101, replace=TRUE)  # simulated coin tosses
y100 = 5 + filter(x, sides=1, filter=c(1,-.5))[-1] 
y10  = y100[1:10]
tsplot(y10, type='s', col=4, yaxt='n', xaxt='n', gg=TRUE)  
 axis(1, 1:10); axis(2, seq(2,8,2), las=1)
 points(y10, pch=21, bg=6)    
round( acf1(y10, 4, plot=FALSE), 2)   #  1/\sqrt{10}  =.32
round( acf1(y100, 4, plot=FALSE), 2)  #  1/\sqrt{100} =.1

```

<br/> Example 1.28

```r
acf1(speech, 250, col=4)

```

<br/> Example 1.29

```r
par(mfrow=c(3,1))
acf1(soi, 48, main="Southern Oscillation Index")
acf1(rec, 48, main="Recruitment")
ccf2(soi, rec, 48, main="SOI vs Recruitment")

```

<br/> Example 1.30

```r
set.seed(90210)
num = 250  
t   = 1:num
X   = .02*t + rnorm(num,0,2)
Y   = .01*t + rnorm(num)
par(mfrow=c(3,1))
tsplot(cbind(X,Y), col=c(4,6), ylab="data", spaghetti=TRUE, lwd=2, gg=TRUE)
ccf2(X, Y, ylim=c(-.4,.5), col=4, lwd=2, gg=TRUE)
ccf2(X, detrend(Y), ylim=c(-.4,.5), col=4, lwd=2, gg=TRUE)

```

<br/> Example 1.31

```r
persp(1:64, 1:36, soiltemp, phi=25, theta=25, scale=FALSE, expand=4, ticktype="detailed", xlab="rows", ylab="cols", zlab="temperature", col="lightblue")

dev.new()
tsplot(rowMeans(soiltemp), xlab="row", ylab="Average Temperature")

```

<br/> Example 1.32

```r
fs = abs(fft(soiltemp-mean(soiltemp)))^2/(64*36) # see Ch 4 for info on FFT
cs = Re(fft(fs, inverse=TRUE)/sqrt(64*36))  # ACovF
rs = cs/cs[1,1]                             # ACF

rs2 = cbind(rs[1:41,21:2], rs[1:41,1:21])   #  these lines are just to center
rs3 = rbind(rs2[41:2,], rs2)                #  the 0 lag  

par(mar = c(1,2.5,0,0)+.1)
persp(-40:40, -20:20, rs3, phi=30, theta=30, expand=30, scale="FALSE", ticktype="detailed", xlab="row lags", ylab="column lags", zlab="ACF", col="lightblue")

```

<br/> Bad LCG

```r
x = c(1)  # set the seed to 1
for (n in 2:24){ x[n] = (5*x[n-1] + 2) %% (2^4) }
x         # print x

```

[<sub>top</sub>](#table-of-contents)

---



## Chapter 2


<br/> Example 2.1

```r
par(mfrow=2:1)
trend(chicken, lwd=2, results=TRUE) # graphic and results
trend(salmon, lwd=2)                # graphic only

```

<br/> Example 2.2

```r
par(mfrow = c(3,1))
tsplot(cmort, ylab="Rate per 10,000", type="o", pch=19, col=6, nxm=2, main="Cardiovascular Mortality")
tsplot(tempr, ylab="\u00B0F", type="o", pch=19, col=4, nxm=2, main="Temperature")
tsplot(part, ylab="PPM", type="o", pch=19, col=2, nxm=2, main="Particulates")
dev.new()
pairs(cbind(Mortality=cmort, Temperature=tempr, Particulates=part), col=4, lower.panel = astsa:::.panelcor)
temp = tempr - mean(tempr)  # center temperature
temp2 = temp^2
trend = time(cmort)
fit = lm(cmort~ trend + temp + temp2 + part, na.action=NULL)
summary(fit)  # regression results
summary(aov(fit))  # ANOVA table (compare to n<br/> Ext line)
summary(aov(lm(cmort~cbind(trend, temp, temp2, part)))) # Table 2.1
num = length(cmort)  # sample size
AIC(fit)/num - log(2*pi)  # AIC as in (2.15)
BIC(fit)/num - log(2*pi)  # BIC as in (2.17)
(AICc = log(sum(resid(fit)^2)/num) + (num+5)/(num-5-2)) # AICc

```

<br/> Example 2.3

```r
# uses variables from previous example
summary(fit2 <- lm(cmort~ trend + temp + temp2 + part + co, data=lap, na.action=NULL))
# compare models
c( AIC(fit),  BIC(fit))/num   # model without co
c( AIC(fit2), BIC(fit2))/num  # model with co

```



<br/> Example 2.4  

First, the Lotka-Volterra simulation (code not in the book)

```R
H = c(1); L =c(.5)
for (t in 1:66000){
H[t+1] = 1.0015*H[t] - .00060*L[t]*H[t] 
L[t+1]  = .9994*L[t] + .00025*L[t]*H[t]
}
L = ts(10*L, start=1850, freq=900)
H = ts(10*H, start=1850, freq=900)

tsplot(cbind(H,L), spag=T, col=c(2,4), ylim=c(0,134), ylab="Population Size", gg=TRUE)
legend('topleft', legend=c('predator', 'prey'), lty=1, col=c(4,2), bty='n', horiz=TRUE, cex=.9)

```

and now back to our regularly scheduled program...

```R
prdtr = ts.intersect(L=Lynx, L1=lag(Lynx,-1), H1=lag(Hare,-1), dframe=TRUE)
summary( fit <- lm(L~ L1 + L1:H1, data=prdtr, na.action=NULL) )

# residuals
par(mfrow=1:2)
tsplot(resid(fit), col=4, main="")
acf1(resid(fit), col=4, main="")
mtext("Lynx Residuals", outer=TRUE, line=-1.4, font=2)

# using dynlm
library(dynlm)
summary( fit2 <- dynlm(Lynx~ L(Lynx,1) + L(Lynx,1):L(Hare,1)) )

```




<br/> Example 2.6

```r
par(mfrow=2:1)
tsplot(detrend(chicken), col=4, main="detrended" )
tsplot(diff(chicken), col=4, main="first difference")

dev.new()
par(mfrow = c(3,1))
acf1(chicken, col=6, lwd=2)
acf1(detrend(chicken), col=3, lwd=2)
acf1(diff(chicken), col=4, lwd=2)

```

<br/> Example 2.7

```r
par(mfrow = 2:1)
tsplot(diff(gtemp_land), col=4, xlab="Year")
acf1(diff(gtemp_land), col=4)
mean(diff(window(gtemp_land, end=1980)))   # drift until 1980
mean(diff(window(gtemp_land, start=1980))) # drift since 1980

```



<br/> Example 2.8  

```r
layout(matrix(1:4,2), widths=c(2.5,1))
tsplot(varve, main="", ylab="", col=4)
 mtext("varve", side=3, line=.5, cex=1.2, font=2, adj=0)
tsplot(log(varve), main="", ylab="", col=4)
 mtext("log(varve)", side=3, line=.5, cex=1.2, font=2, adj=0)

# Some OSs (think macOS) don't play with panel.first, so remove it if necessary
qqnorm(varve, main=NA, col=4, panel.first=Grid(minor=FALSE)); qqline(varve, col=2, lwd=2)
qqnorm(log(varve), main=NA, col=4, panel.first=Grid(minor=FALSE)); qqline(log(varve), col=2, lwd=2) 

```


<br/> Example 2.9

```r
lag1.plot(soi, 12, col=4)      # Figure 2.10
dev.new()
lag2.plot(soi, rec, 8, col=4)  # Figure 2.11

```

<br/> Example 2.10

```r
dummy = ifelse(soi<0, 0, 1)
fish = ts.intersect(R=rec, SL6=lag(soi,-6), DL6=lag(dummy,-6), dframe=TRUE)
summary(fit <- lm(R~ SL6*DL6, data=fish, na.action=NULL))
layout(matrix(1:2,2), heights = c(3,2))
tsplot(fish[,"SL6"], fish[,"R"], type="p", col=8, xlab=bquote(S[~t-6]), ylab=bquote(R[~t]))
lines(lowess(fish[,"SL6"], fish[,"R"]), col=4, lwd=2)
points(fish[,"SL6"], fitted(fit), pch="+", col=2)
tsplot(resid(fit), col=4)

```

 

<br/> Example 2.11

```r
set.seed(90210) 
t  = 1:500
x  = 2*cos(2*pi*(t+15)/50) + rnorm(500,0,5)
z1 = cos(2*pi*t/50)  
z2 = sin(2*pi*t/50)
summary(fit <- lm(x~0+z1+z2))  # zero to exclude the intercept
par(mfrow=c(2,1))
tsplot(x, col=4, gg=TRUE)
tsplot(x, ylab=bquote(hat(x)), col=4, gg=TRUE)
lines(fitted(fit), col=2, lwd=2)

```


<br/> Example 2.12

```r
set.seed(90210)
t = 1:500
x = 2*cos(2*pi*(t+15)/50) + rnorm(500,0,5)
acf1(x, 200)
summary(fit <- nls(x~ A*cos(2*pi*omega*t + phi), start=list(A=10, omega=1/55, phi=0)))
tsplot(x, ylab=bquote(hat(x)), col=4, gg=TRUE)
lines(fitted(fit), col=2, lwd=2)

```



<br/> Example 2.13

```r
wgts = c(.5, rep(1,11), .5)/12
ENSOf = filter(ENSO, sides=2, filter=wgts)
tsplot(ENSO, col=8)
lines(ENSOf, lwd=2, col=4)
par(fig = c(.02, .25, .01, .4), new=TRUE, bty="n")
nwgts = c(rep(0,6), wgts, rep(0,6))
plot(nwgts, type="l", xaxt="n", yaxt="n", ann=FALSE)

```



<br/> Example 2.14 

```r
tsplot(ENSO, col=8)
lines(ksmooth(time(ENSO), ENSO, "normal", bandwidth=1), lwd=2, col=4)
par(fig = c(.02, .25, .01, .4), new=TRUE, bty="n")
curve(dnorm,-4,4, xaxt="n", yaxt="n", ann=FALSE)

```


<br/> Example 2.15

```r
trend(ENSO, lowess=TRUE, col=c(8,6))  # data and trend
lines(lowess(ENSO, f=.03), lwd=2, col=4)  # El Niño cycle

```



<br/> Example 2.16

```r
trend(ENSO, order=3)  # not shown

tsplot(ENSO, col=8)
lines(smooth.spline(time(ENSO), ENSO, spar= 1), lwd=2, col=6)  # trend
lines(smooth.spline(time(ENSO), ENSO, spar=.5), lwd=2, col=4)  # El Niño

```

<br/> Example 2.17

```r
x = window(hor, start=2002)
plot(decompose(x))  # not shown
plot(stl(x, s.window="per")) # seasons are perfectly periodic - not shown
plot(stl(x, s.window=15))

# better graphic
par(mfrow = c(4,1))
x = window(hor, start=2002)
out = stl(x, s.window=15)$time.series
tsplot(x, main="Hawaiian Occupancy Rate", ylab="% rooms", col=8, type="c")
text(x, labels=1:4, col=c(3,4,2,6), cex=1.25)
tsplot(out[,1], main="Seasonal", ylab="% rooms", col=8, type="c")
text(out[,1], labels=1:4, col=c(3,4,2,6), cex=1.25)
tsplot(out[,2], main="Trend", ylab="% rooms", col=8, type="c")
text(out[,2], labels=1:4, col=c(3,4,2,6), cex=1.25)
tsplot(out[,3], main="Noise", ylab="% rooms", col=8, type="c")
text(out[,3], labels=1:4, col=c(3,4,2,6), cex=1.25)

```

[<sub>top</sub>](#table-of-contents)

---



## Chapter 3


<br/> Example 3.2

```r
par(mfrow=2:1)
tsplot(sarima.sim(ar= .9, n=100), ylab="x", col=4, gg=TRUE, main=bquote(AR(1)~~~phi==+.9))
tsplot(sarima.sim(ar=-.9, n=100), ylab="x", col=4, gg=TRUE, main=bquote(AR(1)~~~phi==-.9))

```

<br/> Example 3.5

```r
par(mfrow = 2:1)
tsplot(sarima.sim(ma= .9, n=100), ylab="x", col=4, gg=TRUE, main=bquote(MA(1)~~~phi==+.9))
tsplot(sarima.sim(ma=-.9, n=100), ylab="x", col=4, gg=TRUE, main=bquote(MA(1)~~~phi==-.9))  

```

<br/> Example 3.7

```r
set.seed(8675309)         # Jenny, I got your number
x = rnorm(150, mean=5)    # Jenerate iid N(5,1)s
arima(x, order=c(1,0,1))  # Jenstimation

```

<br/> Example 3.8

```r
ARMAtoMA(ar = .9, ma = .5, 10)   # first 10 psi-weights
ARMAtoAR(ar = .9, ma = .5, 10)   # first 10 pi-weights
ARMAtoMA(ar=1, ma=0, 20)

```

<br/> Example 3.9

```r
# this is how Figure 3.3 was generated
seg1   =  seq( 0, 2,  by=0.1)
seg2   =  seq(-2, 2,  by=0.1)
name1  =  bquote(phi[1])
name2  =  bquote(phi[2])
tsplot(seg1, (1-seg1), ylim=c(-1,1), xlim=c(-2,2), ylab=name2, xlab=name1, main='Causal Region of an AR(2)')
 lines(-seg1, (1-seg1), ylim=c(-1,1), xlim=c(-2,2)) 
 abline(h=0, v=0, lty=2, col=8)
 lines(seg2, -(seg2^2 /4), ylim=c(-1,1))
 lines(x=c(-2,2), y=c(-1,-1), ylim=c(-1,1))
 text(0, .35, 'real roots')
 text(0, -.5, 'complex roots')

```


<br/> Example 3.11

```r
ARMAtoMA(ar=.9, ma=.5, 50)  # for a list
plot(ARMAtoMA(ar=.9, ma=.5, 50))  # for a graph

```



<br/> Example 3.12

```r
set.seed(8675309)
x = sarima.sim(ar=c(1.5,-.75), n=144, S=12)
psi = ts(c(1, ARMAtoMA(ar=c(1.5, -.75), ma=0, 50)), start=0, freq=12)
par(mfrow=c(2,1))
tsplot(x, col=4, xaxt="n", gg=TRUE, main=bquote(AR(2)~~~phi[1]==1.5~~~phi[2]==-.75))
mtext(seq(0,144,by=12), side=1, at=0:12, cex=.8)
tsplot(psi, col=4, type="o", xaxt="n", gg=TRUE, xlab="Index", ylab=bquote(psi-weights))
mtext(seq(0,48,by=12), side=1, at=0:4, cex=.8)

# roots of the polynomial 
z = c(1,-1.5,.75)  # coefficients of the polynomial
(a = polyroot(z)[1]) # print one root = 1 + i/sqrt(3)
Arg(a) # in radians/pt
(theta = Arg(a)/(2*pi)) # in cycles/pt
1/theta # the pseudo period

```

<br/> Example 3.15

```r
ACF  = ts(ARMAacf(ar=c(1.5,-.75), lag=24), start=0, freq=12)
PACF = ts(c(NA, ARMAacf(ar=c(1.5,-.75), lag=24, pacf=TRUE)), start=0, freq=12)
par(mfrow=1:2)
tsplot(ACF, type="h", xlab="LAG", ylim=c(-.8,1), gg=TRUE, col=4, xaxt="n")
abline(h=0, col=8)
mtext(side=1, at=seq(0,2,by=.5), text=seq(0,24,by=6), cex=.8)
tsplot(PACF, type="h", xlab="LAG", ylim=c(-.8,1), gg=TRUE, col=4, xaxt="n")
abline(h=0, col=8)
mtext(side=1, at=seq(0,2,by=.5), text=seq(0,24,by=6), cex=.8)

```

<br/> Example 3.17  

```r
acf2(rec, 48)     # will produce values and a graphic 
(regr = ar.ols(rec, order=2, demean=FALSE, intercept=TRUE))  # regression
regr$asy.se.coef  # standard errors 

```


<br/> Example 3.23  

```r
regr = ar.ols(rec, order=2, demean=FALSE, intercept=TRUE)
fore = predict(regr, n.ahead=24)
x = ts( c(rec, fore$pred), start=1950, frequency=12)
tsplot(window(x, start=1980), ylab="Recruitment", ylim=c(10,100))
lines(fore$pred, type="o", col=2)
U = fore$pred+fore$se
L = fore$pred-fore$se
xx = c(time(U), rev(time(U)))
yy = c(L, rev(U))
polygon(xx, yy, border = 8, col = gray(0.6, alpha = 0.2))

```



<br/> Example 3.25

```r
set.seed(1984)
x = sarima.sim(ar=.9, ma=.5, n=100)  # simulate
xr = rev(x)  # reverse data
pxr = sarima.for(xr,10,1,0,1, plot=FALSE) # backcast
pxrp = rev(pxr$pred) # reorder the predictors (for plotting)
pxrse = rev(pxr$se) # reorder the SEs
nx = ts(c(pxrp, x), start=-9)  # attach the backcasts to the data
tsplot(nx, ylab=bquote(X[~t]), main="Backcasting", ylim=c(-5,4), col=4, gg=TRUE)
U = nx[1:10] + pxrse
L = nx[1:10] - pxrse
xx = c(-9:0, 0:-9)
yy = c(L, rev(U))
polygon(xx, yy, border = 8, col = gray(0.6, alpha = 0.2))
lines(-9:0, nx[1:10], col=2, type="o")

```

<br/> Example 3.27

```r
rec.yw = ar.yw(rec, order=2)
rec.yw$x.mean    # = 62.26278 (mean estimate)
rec.yw$ar        # = 1.3315874, -.4445447  (parameter estimates)
sqrt(diag(rec.yw$asy.var.coef))  # = .04222637, .04222637  (standard errors)
rec.yw$var.pred  # = 94.79912 (error variance estimate)

rec.pr = predict(rec.yw, n.ahead=24)
tsplot(cbind(rec, rec.pr$pred), col=1:2, spaghetti=TRUE)
lines(rec.pr$pred + rec.pr$se, col=2, lty=5)
lines(rec.pr$pred - rec.pr$se, col=2, lty=5)

```

<br/> Example 3.28

```r
# generate 10000 MA(1)s and calculate the 1st sample ACF
x = replicate(10000, acf1(sarima.sim(ma=.9, n=100), max.lag=1, plot=FALSE))  
1 - ecdf(abs(x))(.5)   # .5 exceedance prob (is about 38%)
hist(x); abline(v=.5, col=2)  # for fun (not in text)

# The asymptotic approximation is   (not in text)
pnorm( (.5-.497)/.071, lower=FALSE)  # = 0.4831483
```



<br/> Example 3.30

```r
rec.mle = ar.mle(rec, order=2)
rec.mle$x.mean
rec.mle$ar
sqrt(diag(rec.mle$asy.var.coef))
rec.mle$var.pred

```



<br/> Example 3.32

```r
acf2(diff(log(varve)), col=4)  # sample ACF and PACF
x = diff(log(varve))       # data
r = acf1(x, 1, plot=FALSE) # acf(1)
c(0) -> z -> Sc -> Sz -> Szw -> para # initialize .. 
c(x[1]) -> w                         # .. all variables
num = length(x)            # 633

## Gauss-Newton Estimation
para[1] = (1-sqrt(1-4*(r^2)))/(2*r)  # MME to start (not very good)
niter   = 12             
for (j in 1:niter){  
 for (t in 2:num){ 
   w[t] = x[t]   - para[j]*w[t-1]
   z[t] = w[t-1] - para[j]*z[t-1]
 }
 Sc[j]  = sum(w^2)
 Sz[j]  = sum(z^2)
 Szw[j] = sum(z*w)
para[j+1] = para[j] + Szw[j]/Sz[j]
}
## Results
cbind(iteration=1:niter-1, thetahat=para[1:niter], Sc, Sz)

## Plot conditional SS and results
c(0) -> cSS
th = -seq(.3, .94, .01)
for (p in 1:length(th)){   
 for (t in 2:num){ w[t] = x[t] - th[p]*w[t-1] 
 }
cSS[p] = sum(w^2)
}
tsplot(th, cSS, ylab=bquote(S[c](theta)), xlab=bquote(theta))
abline(v=para[1:12], lty=2, col=4) # add previous results to plot
points(para[1:12], Sc[1:12], pch=16, col=4)

```



<br/> Example 3.34

```r
t = time(USpop) - 1955
reg = lm( USpop~ t+I(t^2)+I(t^3)+I(t^4)+I(t^5)+I(t^6)+I(t^7)+I(t^8) )
b = as.vector(reg$coef)
g = function(t){ b[1] + b[2]*(t-1955) + b[3]*(t-1955)^2 + b[4]*(t-1955)^3 + b[5]*(t-1955)^4 + b[6]*(t-1955)^5 + b[7]*(t-1955)^6 + b[8]*(t-1955)^7 + b[9]*(t-1955)^8
}
x = 1900:2024
tsplot(x, g(x), ylab="Population", xlab="Year", main="U.S. Population by Official Census", cex.main=1, col=4)
points(time(USpop), USpop, pch=21, bg=rainbow(12), cex=1.25)
mtext(bquote("\u00D7"~10^6), side=2, line=1.5, adj=1, cex=.8)

```


<br/> Example 3.35

```r
# data
set.seed(101010)
e   = rexp(150, rate=.5); u = runif(150,-1,1); de = e*sign(u)  
dex = 50 + sarima.sim(n=100, ar=.95, innov=de, burnin=50)
layout(matrix(1:2, nrow=1), widths=c(5,2))
tsplot(dex, col=4, ylab=bquote(X[~t]), gg=TRUE)
# densities
f = function(x) { .5*dexp(abs(x), rate = 1/sqrt(2))}
w = seq(-5, 5, by=.01)
tsplot(w, f(w), gg=TRUE, col=4, xlab='w', ylab='f(w)', ylim=c(0,.4)) 
lines(w, dnorm(w), col=2) 

fit = ar.yw(dex, order=1, aic=FALSE)
round(estyw <- c(mean=fit$x.mean, ar1=fit$ar, se=sqrt(fit$asy.var.coef), var=fit$var.pred), 3)

set.seed(111)
phi.yw = c()
for (i in 1:1000){
  e  = rexp(150, rate=.5)
  u  = runif(150,-1,1)
  de = e*sign(u)
  x  = 50 + sarima.sim(n=100, ar=.95, innov=de, burnin=50)
  phi.yw[i] = ar.yw(x, order=1)$ar   
} 


# Bootstrap
boots = ar.boot(dex, order=1, plot=FALSE)  # default is B = 500
phi.star.yw = boots[[1]]       # bootstrapped phi  
# Picture
dev.new()
hist(phi.star.yw, main=NA, prob=TRUE, xlim=c(.65,1.05), ylim=c(0,15), col=astsa.col(4,.4), xlab=bquote(hat(phi)), breaks="FD")
lines(density(phi.yw, bw=.02), lwd=2) # from previous simulation
u = seq(.75, 1.1, by=.001)            # normal approximation
lines(u, dnorm(u, mean=estyw[2], sd=estyw[3]), lty=2, lwd=2)
legend(.65, 15, bty="n", lty=c(1,0,2), lwd=c(2,0,2), col=1, pch=c(NA,22,NA), pt.bg=c(NA,astsa.col(4,.4),NA), pt.cex=2.5, legend=c("true distribution",   "bootstrap distribution", "normal approximation"))

# 95% CI
alf = .025
quantile(phi.star.yw, probs = c(alf, 1-alf))      # boot
quantile(phi.yw, probs = c(alf, 1-alf))           # true
qnorm(c(alf, 1-alf), mean=estyw[2], sd=estyw[3])  # asym normal

```




<br/> Example 3.36

```r
set.seed(1234567)  
x = ts(cumsum(rnorm(150, .2)))  # RW with drift .2 and error sd 1
y = window(x, end=100)          # first 100 obs
c(d <- mean(diff(y)), s <- sd(diff(y))) # estimated drift and error sd
rmspe = s*sqrt(1:50)
yfore = ts(y[100] + 1:50*d, start=101)
tsplot(x, ylab=bquote(X[~t]), col=4, gg=TRUE, ylim=c(0,25))
lines(yfore, col=6)
  xx = c(101:150, 150:101)
  yy = c(yfore - 1*rmspe, rev(yfore + 1*rmspe))
polygon(xx, yy, border = NA, col = gray(0.6, alpha = 0.2))
text(85, 23, 'PAST', cex=.8); text(115, 23, 'FUTURE', cex=.8) 
abline(v=100, lty=2) 

```

<br/> Example 3.37

```r
set.seed(666)
x = sarima.sim(d = 1, ma = -0.8, n = 100)
(x.ima = HoltWinters(x, beta=FALSE, gamma=FALSE))
plot(x.ima)

```

<br/> Example 3.38

```r
sarima(log(varve), 0, 1, 1, col=4)
sarima(log(varve), 1, 1, 1, no.constant=TRUE, col=4)

```

<br/> Example 3.39

```r
trend = time(cmort); temp = tempr - mean(tempr); temp2 = temp^2
summary(fit <- lm(cmort~trend + temp + temp2 + part, na.action=NULL))
acf2(resid(fit), 52)   # implies AR2
sarima(cmort, 2,0,0, xreg=cbind(trend, temp, temp2, part))  

```



<br/> Example 3.40

```r
pp = ts.intersect(L=Lynx, L1=lag(Lynx,-1), H1=lag(Hare,-1), dframe=TRUE)
# Original Regression
summary( fit <- lm(L~ L1 + L1:H1, data=pp, na.action=NULL) )
acf2(resid(fit), col=4)   # ACF/PACF of the residuls
# Try AR(2) errors
sarima(pp$L, 2,0,0, xreg=cbind(L1=pp$L1, LH1=pp$L1*pp$H1), col=4)

```



<br/> Example 3.41

```r
set.seed(10101010)
SAR = sarima.sim(sar=.95, S=12, n=37) + 50
layout(matrix(c(1,2, 1,3), nc=2), heights=c(1.5,1))
tsplot(SAR, type="c", xlab="Year", gg=TRUE, ylab='SAR(1)', xaxt='n')
 abline(v=0:3, col=4, lty=2)
 points(SAR, pch=Months, cex=1.2, font=4, col=1:6)
 axis(1, at=0:3, col='white')
phi  = c(rep(0,11),.95)
ACF  = ARMAacf(ar=phi, ma=0, 100)[-1] # [-1] removes 0 lag
PACF = ARMAacf(ar=phi, ma=0, 100, pacf=TRUE)
LAG  = 1:100/12
tsplot(LAG, ACF, type="h", xlab="LAG \u00F7 12", ylim=c(-.04,1), gg=TRUE, col=4)
 abline(h=0, col=8)
tsplot(LAG, PACF, type="h", xlab="LAG \u00F7 12", ylim=c(-.04,1), gg=TRUE, col=4)
 abline(h=0, col=8)

```

<br/> Example 3.42

```r
phi  = c(rep(0,11),.8)
ACF  = ts(ARMAacf(ar=phi, ma=-.5, 50), start=0, freq=12)     
PACF = ts(c(0, ARMAacf(ar=phi, ma=-.5, 50, pacf=TRUE)), start=0, freq=12)
par(mfrow=1:2)
tsplot(ACF,  type="h", xlab="LAG \u00F7 12", gg=TRUE, col=4)  
 abline(h=0, col=8)
tsplot(PACF, type="h", xlab="LAG \u00F7 12", gg=TRUE, col=4)  

dev.new()
tsplot(gtemp.month, spaghetti=TRUE, col=rainbow(49, start=.2, v=.8, rev=TRUE), ylab='\u00b0C', xlab='Month', xaxt='n', main='Mean Monthly Global Temperature')
axis(1, labels=Months, at=1:12)
lines(gtemp.month[,1],  lwd=2, col=6)
lines(gtemp.month[,49], lwd=2, col=3)
text(10, 13, '1975')
text(10.3, 15.5, '2023')

```

<br/> Example 3.44

```r
par(mfrow=2:1)
tsplot(cardox, col=4, ylab=bquote(CO[2]), main="Monthly Carbon Dioxide Readings - Mauna Loa Observatory")
tsplot(diff(diff(cardox,12)), col=4, ylab=bquote(nabla~nabla[12]~CO[2]))
acf2(diff(diff(cardox,12)), col=4) 
sarima(cardox, p=0,d=1,q=1, P=0,D=1,Q=1,S=12, col=4)
sarima(cardox, 1,1,1, 0,1,1,12)

sarima.for(cardox, 60, 1,1,1, 0,1,1,12, col=4)
abline(v=2023.17, lty=6)
##-- for comparison, try the first model --##
sarima.for(cardox, 60, 0,1,1, 0,1,1,12)  # not shown 

```

 

[<sub>top</sub>](#table-of-contents)

---



## Chapter 4

Aliasing

```r
t = seq(0, 24, by=.1)
X = cos(2*pi*t/2)                # one cycle every 2 hrs
tsplot(t, X, xlab="Hours", ylab=bquote(X[~t]), gg=TRUE, col=7)
T = seq(1, length(t), by=25)    # observe every 2.5 hrs 
points(t[T], X[T], pch=19, col=4)
lines(t, cos(2*pi*t/10), col=4)

```

<br/> Example 4.1

```r
x1 = 2*cos(2*pi*1:100*6/100)  + 3*sin(2*pi*1:100*6/100)
x2 = 4*cos(2*pi*1:100*10/100) + 5*sin(2*pi*1:100*10/100)
x3 = 6*cos(2*pi*1:100*40/100) + 7*sin(2*pi*1:100*40/100)
x  = x1 + x2 + x3
par(mfrow = c(2,2), cex.main=1, font.main=1)
tsplot(x1, ylim=c(-10,10), main=bquote(omega==6/100~~A^2==13),  col=4, gg=TRUE)
tsplot(x2, ylim=c(-10,10), main=bquote(omega==10/100~~A^2==41), col=4, gg=TRUE)
tsplot(x3, ylim=c(-10,10), main=bquote(omega==40/100~~A^2==85), col=4, gg=TRUE)
tsplot(x,  ylim=c(-16,16), main="sum", col=4, gg=TRUE)

```

<br/> Example 4.2

```r
# x from previous example used here
per = Mod( fft(x)/sqrt(100) )^2
P = (4/100)*per;  Fr = 0:99/100
tsplot(Fr, P, type="h", lwd=3, xlab="frequency", ylab="scaled periodogram", col=4, gg=TRUE)
abline(v=.5, lty=5, col=8)

```

<br/> Example 4.4

```r
par(mfrow=c(3,2))
for(i in 4:9){
mvspec(fmri1[,i], main=colnames(fmri1)[i], ylim=c(0,3), xlim=c(0,.2), col=5, lwd=2, type='o', pch=20)
abline(v=1/32, col=4, lty=5)  # stimulus frequency
}

```

<br/> Examples 4.5

```r
par(mfrow=2:1)
t = 1:200
tsplot(x <- 2*cos(2*pi*.2*t)*cos(2*pi*.01*t))   # not shown
lines(cos(2*pi*.19*t)+cos(2*pi*.21*t), col=2)   # the same
Px = mvspec(x, main='')                         # the periodogram

par(mfrow=2:1)
tsplot(star, ylab="star magnitude", xlab="day", col=4)
Pstar = mvspec(star, col=5, xlim=c(0,.08), lwd=3, type="h", main=NA)
text(.05, 7000, "24 day cycle"); text(.027, 9000, "29 day cycle")
Pstar$details[19:26,]

```


<br/> Example 4.9

```r
round(z <- polyroot(c(1,-1,.9)), 3)
Arg(z[1])/(2*pi)

par(mfrow=c(3,1))
arma.spec(main="White Noise", col=5, gg=TRUE)
arma.spec(ma=.9, main="Moving Average", col=5, gg=TRUE)
arma.spec(ar=c(1,-.9), main="Autoregression", col=5, gg=TRUE)

```

<br/> DFT  - it's injective

```r
( dft = fft(1:4)/sqrt(4) )
( idft = fft(dft, inverse=TRUE)/sqrt(4) )
( Re(idft) )  # keep it real

```


<br/> Example 4.12

```r
x = c(1, 2, 3, 2, 1);  t=1:5
omega1 = cbind(cos(2*pi*t*1/5), sin(2*pi*t*1/5))
omega2 = cbind(cos(2*pi*t*2/5), sin(2*pi*t*2/5))
anova(lm(x~ omega1 + omega2))    # ANOVA Table
Mod(fft(x))^2/5       # the periodogram (as a check)

```



<br/> Example 4.15

```r
P = mvspec(ENSO, lowess=TRUE, col=5, main='ENSO: Raw Periodogram')
 rect(1/7,-1, 1/2, 4, density=NA, col=gray(.6,.2))
 abline(v=1/4, lty=5, col=8)
 mtext('1/4',side=1, line=0, at=.25, cex=.75)
# confidence interval:
c(2*P$spec[18]/qchisq(.975, 2),  2*P$spec[18]/qchisq(.025, 2))

```

<br/> smoothing the periodogram

```r
P = mvspec(rnorm(2^10), col=8, main=NA, ylab='periodogram', gg=TRUE)
segments(0,1, .5,1, col=astsa.col(6,.7), lwd=5)  # actual spectrum
lines(P$freq, filter(P$spec, filter=rep(.01,100), circular=TRUE), col=4, lwd=3)

```




<br/> Example 4.16

```r
kd = kernel("daniell", 4)  # nine 1/9s
par(mfrow=2:1)
ENSO.av  = mvspec(ENSO, lowess=TRUE, kernel=kd, col=5, main='ENSO: Averaged Periodogram')
 rect(1/7,-1, 1/2,4, density=NA, col=gray(.6,.2))
 abline(v=1/4, lty=5, col=8)
 mtext('1/4', side=1, line=0, at=.25, cex=.75)
ENSO.avl = mvspec(ENSO, lowess=TRUE, kernel=kd, col=5, main='ENSO: Averaged Periodogram (log scale)', log='y')
 rect(1/7, .005, 1/2, 1, density=NA, col=gray(.6,.2))
 abline(v=1/4, lty=5, col=8)
 mtext('1/4', side=1, line=0, at=.25, cex=.75)

```



<br/> Example 4.17

```r
y = ts(100:1 %% 20, freq=20)   # sawtooth signal
par(mfrow=2:1)
tsplot(1:100, y, ylab='sawtooth signal', col=4, gg=TRUE)
mvspec(y, main=NA, ylab='periodogram', col=5, gg=TRUE)

```

<br/> Modified Daniell kernel

```r
par(mfrow=1:2)
tsplot(kernel("modified.daniell", c(3,3)), ylab=bquote(h[~k]), lwd=2, col=4, ylim=c(0,.16), xlab='k', type='h', main='mDaniell(3,3)', gg=TRUE)
tsplot(kernel("modified.daniell", c(3,3,3)), ylab=bquote(h[~k]), lwd=2, col=4, ylim=c(0,.16), xlab='k', type='h', main='mDaniell(3,3,3)', gg=TRUE)

```



<br/> Example 4.18

```r
par(mfrow=2:1)
ENSO.sm = mvspec(ENSO, lowess=TRUE, spans=c(7,7), col=5, main='ENSO: Smoothed Periodogram')
 rect(1/7, -1, 1/2, 4, density=NA, col=gray(.6,.2))
 abline(v=1/4, lty=5, col=8)
 mtext('1/4',side=1, line=0, at=.25, cex=.75)
ENSO.sml = mvspec(ENSO, lowess=TRUE, spans=c(7,7), col=5, main='ENSO: Smoothed Periodogram (log scale)', log='y')
 rect(1/7, .005, 1/2,4, density=NA, col=gray(.6,.2))
 abline(v=1/4, lty=5, col=8)
 mtext('1/4',side=1, line=0, at=.25, cex=.75)

```




<br/> Example 4.19

```r
mvspec(ENSO, lowess=TRUE, spans=c(7,7,7), taper=.5, xlim=c(0,3), col=5)      
s0 = mvspec(ENSO, lowess=TRUE, spans=c(7,7,7), plot=FALSE)   # no taper
lines(s0$freq, s0$spec, col=2, lty=5) 
text(.22, .4, 'leakage', cex=.8)
legend('bottomleft', legend=c('no taper', 'full taper'), lty=c(5,1), col=c(2,4), bty='n')

```


<br/> Example 4.20

```r
par(xpd = NA, oma=c(0,0,0,5)) 
x = rep(1,100) 
tsplot(1:100/100, cbind(spec.taper(x, p=.1), spec.taper(x, p=.2), spec.taper(x, p=.5)), col=astsa.col(2:4,.5), lty=c(5,2,1), gg=TRUE, spaghetti=TRUE, xlab='t / n', lwd=2, ylab='taper')
legend('topright', inset=c(-.15,0), bty='n', lty=c(5,2,1), col=2:4, legend=c('10%','20%', 'Full'), lwd=2)

```


<br/> Example 4.21

```r
par(mfrow=2:1)
mvspec(ts(scale(EQ5), freq=40), spans=c(21,21), xlim=c(0,10), taper=.1, col=5, main='Earthquake', xlab='frequency (Hz)')
mvspec(ts(scale(EXP6), freq=40), spans=c(21,21), xlim=c(0,10), taper=.1, col=5, main='Explosion', xlab='frequency (Hz)')

```



<br/> Example 4.22

```r
spec.ic(ENSO, lowess=TRUE, col=astsa.col(5, .7), ylim=c(0,.65), lwd=2)
u = mvspec(ENSO, lowess=TRUE, spans=c(7,7), taper=.2, plot=FALSE)
lines(u$freq, u$spec, col=6, lty=5)
legend('topright', legend=c('Parameteric', 'Nonparametric'), lty=c(1,5), col=5:6, bg='white')

```


<br/> Example 4.25

```r
sr = mvspec(cbind(soi,rec), kernel=kernel("daniell",9), col=5, ci.col=8, ci.lty=2, plot.type='coh', main='SOI & Recruitment') 
f = qf(.999, 2, sr$df-2)  
abline(h = f/(18+f), col=8)

```

<br/> Example 4.26

```r
par(mfrow=c(3,1))
tsplot(ENSO, main='SOI', col=4, ylab='' )  
tsplot(diff(ENSO), col=4, ylab='', main='First Difference') 
 k = kernel("modified.daniell", 6)  
tsplot(kernapply(ENSO, k), col=4, ylab='', main='Seasonal Moving Average')  
##-- frequency responses --##
w =  seq(0, .5, by=.001) 
FRdiff = abs(1-exp(2i*pi*w))^2
par(mfrow=2:1)
tsplot(12*w, FRdiff, col=4, ylab='', xlab='frequency (\u00D7 12)', main='First Difference')
u = rowSums(cos(outer(w, 2*pi*1:5)))
FRma = ((1 + cos(12*pi*w) + 2*u)/12)^2
tsplot(12*w, FRma, col=4, ylab='', xlab='frequency (\u00D7 12)',  main='Seasonal Moving Average')

```

<br/> Example 4.28

```r
LagReg(soi, rec, L=15, M=32, threshold=6) 
LagReg(rec, soi, L=15, M=32, inverse=TRUE, threshold=.01)

fish = ts.intersect(R=rec, RL1=lag(rec,-1), SL5=lag(soi,-5), dframe=TRUE)
(u = lm(R~ RL1 + SL5, data=fish, na.action=NULL))
acf2(resid(u))  # suggests ar1  
sarima(fish$R, 1, 0, 0, xreg=fish[,2:3]) 

```


<br/> Example 4.29

```r
f.ENSO = SigExtract(ENSO, L=c(21,21), M=64, max.freq=.05)
par(mfrow=2:1)
tsplot(ENSO, col=8)
 lines(f.ENSO, col=4, lwd=2)
mvspec(f.ENSO, lowess=TRUE, spans=c(21,21), taper=.5, col=5, na.action=na.omit)
 rect(1/12,-1, 1/2,1, density=NA, col=gray(.6,.2))
 abline(v=1/3, lty=5, col=8)
 mtext('1/3',side=1, line=0, at=1/3, cex=.75)

```

<br/> Example 4.30

```r
per   = Mod(fft(soiltemp-mean(soiltemp))/sqrt(64*36))^2
 per2 = cbind(per[1:32,18:2], per[1:32,1:18])  # these lines used ...
 per3 = rbind(per2[32:2,], per2)               # ... for better display
persp(-31:31/64, -17:17/36, per3, phi=30, theta=30, expand=.6, ticktype="detailed", xlab="cycles/row", ylab="cycles/column", zlab="Periodogram", col='lightblue')

```


<br/> Example 4.31

> Note: For the remaining examples in this chapter, the breakpoints can vary because GA is random - adjust accordingly

```r
set.seed(90210)
x1 = sarima.sim(ar=c(1.4, -.8), sd=1.5, n=600) 
x2 = sarima.sim(ar=c(1.7, -.8), n=400)
x  = c(x1, x2)
tsplot(x, col=4)
abline(v=600.5, col=2, lwd=2)
autoParm(x)              
ar(x[1:600], order=2)    
ar(x[601:1000], order=2)

mvspec(x)  # all action  < .2 (not displayed)
autoSpec(x, max.freq=.2) 

##-- graphics
z1 = arma.spec(ar=c(1.4, -.8), var=1.5^2, plot=FALSE) 
z2 = arma.spec(ar=c(1.7, -.8), plot=FALSE) 
par(mfrow=2:1)
spec.ic(x1, order=2, main='AutoParm', col=6, gg=TRUE, ylim=c(0,275), xlim=c(0,.25))
 u = spec.ic(x2, order=2, plot=FALSE)
 lines(u[[2]], col=5)
 lines(z2$freq, z2$spec, col=8)
 lines(z1$freq, z1$spec, col=8)
 legend('topright', legend=c('True', 'Segment 1', 'Segment 2'), lty=1, col=c(8,6,5), bty="n")
mvspec(x[598:1000], taper=.5, kernel=bart(3), col=5, main='AutoSpec', gg=TRUE, las=0, xlim=c(0,.25)) 
 u = mvspec(x[1:597], taper=.5, kernel=bart(7), plot=FALSE)
 lines(u$freq, u$spec, col=6, lwd=2)
 lines(z2$freq, z2$spec, col=8)
 lines(z1$freq, z1$spec, col=8)
 legend('topright', legend=c('True', 'Segment 1', 'Segment 2'), lty=1, col=c(8,6,5), bty="n")

```

<br/> Example 4.32

```r
set.seed(90210)
num = 1000
t   = 1:num
w   = 2*pi/25
d   = 2*pi/150
x1  = 2*cos(w*t)*cos(d*t) + rnorm(num)
x2  = cos(w*t) + rnorm(num)
x   = c(x1, x2)
autoParm(x)
spec.ic(x, order=13)   # the chosen estimate (not displayed) 

mvspec(x)  # all action  < .1 (not displayed)
autoSpec(x, max.freq=.1)

#-- graphics  
par(mfrow=c(2,2))
spec.ic(x1, gg=TRUE, col=5, xlim=c(0,.1))     
 segments(x0=.04-1/150, y0=-10, y1=10, col=2)
 segments(x0=.04+1/150, y0=-10, y1=10, col=2)
spec.ic(x2, gg=TRUE, col=5, xlim=c(0,.1))   
 segments(x0=.04, y0=-10, y1 = 5, col=2)
mvspec(x[1:1004], taper=.5, kernel=bart(1), col=5, main='AutoSpec - Segment 1', gg=TRUE, las=0, xlim=c(0,.1)) 
 segments(x0=.04-1/150, y0=-10, y1=10, col=2)
 segments(x0=.04+1/150, y0=-10, y1=10, col=2)
mvspec(x[1005:2000], taper=.5, kernel=bart(1), col=5, main='AutoSpec - Segment 2', gg=TRUE, las=0, xlim=c(0,.1)) 
 segments(x0=.04, y0=-10, y1=10, col=2)

```

<br/> Example 4.33

```r
autoParm(detrend(MEI, lowess=TRUE))  # no breaks found
autoSpec(detrend(MEI, lowess=TRUE), max.freq=1/12) # one break, mid-1979
time(MEI)[354]
x1 = window(detrend(MEI, lowess=TRUE), end=1979.4)
x2 = window(detrend(MEI, lowess=TRUE), start=1979.4)  # June 1979 

#-- graphic
par(mfrow=2:1)
trend(MEI, lowess=TRUE)
mvspec(x1/sd(x1), taper=.2, kernel=bart(2), col=5, lwd=2, main=NA, xlim=c(0,2))
u = mvspec(x2/sd(x2), taper=.2, kernel=bart(2), col=6, plot=FALSE)
lines(u$freq, u$spec, col=6, lwd=2 )
rect(1/7,-1, 1/2,1.5, density=NA, border=NA, col=gray(.6,.2))
abline(v=c(1/1.5,1/2, 1/7, 1/3), lty=5, col=8)
legend('topright', legend=c('1950 - 1979  ', '1979 - 2018  '), lty=1, bg='transparent', bty='n', col=5:6, cex=.9)
mtext('7',  side=1, line=-.2, at=1/7, cex=.75, font=2, col=3)
mtext('3',  side=1, line=-.2, at=1/3, cex=.75, font=2, col=3)
mtext('1.5',side=1, line=-.2, at=2/3, cex=.75, font=2, col=3)
mtext('2',  side=1, line=-.2, at=.5,  cex=.75, font=2, col=3) 

```



[<sub>top</sub>](#table-of-contents)

---

## Chapter 5

Classic long memory (of the way we were &#127926;)

```r
par(mfrow=2:1)
acf1(log(varve), 100) 
acf1(cumsum(rnorm(500)), 100)  

```

<br/> Example 5.1

```r
library(arfima)
summary(varve.fd <- arfima(log(varve)))
innov = resid(varve.fd)[[1]]  
sarima(innov, 0,0,0, no.constant=TRUE, col=4)  # residual analysis  

# plot pi wgts
dev.new()
p = rep(1,31)
for (k in 1:30){ p[k+1] = (k-coef(varve.fd)[1])*p[k]/(k+1) }
tsplot(p[-1], ylab=bquote(pi[j](d)), xlab="Index (j)", type="h", lwd=4, col=2:7, nxm=5)

```


<br/> Example 5.2

```r
library(arfima)
summary(varve1.fd <- arfima(log(varve), order=c(0,0,1)))

```


<br/> Example 5.3

```r
per   = mvspec(log(varve), fast=FALSE, demean=TRUE, plot=FALSE)$spec
n.per = length(per)
m     = floor((n.per)/2  - 1)
d0    = .1
g     = 4*(sin(pi*((1:m)/n.per))^2)
whit.like = function(d){
  g.d      = g^d
  sig2     = (sum(g.d*per[1:m])/m)
  log.like = m*log(sig2) + d*sum(log(g)) + m
  return(log.like)
}
est = optim(d0, whit.like, gr=NULL, method="L-BFGS-B", hessian=TRUE, lower=0, upper=.5)
c(dhat <- est$par, se.dhat <- 1/sqrt(est$hessian), sig2 <- sum(g^dhat*per[1:m])/m)

u    = spec.ic(log(varve), plot=FALSE)  # produces AR(8)
g    = 4*(sin(pi*((1:200)/2000))^2)
fhat = sig2*g^{-dhat}                   # LM spectrum
tsplot(1:200/2000, fhat, log='y', ylim=c(.3,50), ylab="spectrum", xlab="frequency", col=5)
lines(u[[2]][1:100,1], u[[2]][1:100,2], lty=5, col=6)  # AR(8) spectrum

dog = mvspec(log(varve), fast=FALSE, demean=TRUE, plot=FALSE) 
n   = length(varve);  lper = log(dog$spec);  freq = dog$freq
z   = -2*log(2*sin(pi*freq));  m = floor(n^.8)  
summary(lm(lper[1:m]~ z[1:m]))

```


<br/> Example 5.4

```r
library(tseries)
adf.test(log(varve), k=0)               # DF test 
adf.test(log(varve))                    # ADF test 
pp.test(log(varve))                     # PP test 

```



<br/> Example 5.5  

```r
tsplot(diff(log(GNP)), col=4)         # data
acf2(diff(log(GNP)), col=4, main=NA)  # p/acf
library(fGarch)                       # fit ARCH model
summary(gnp.g <- garchFit(~arma(2,0)+garch(1,0), data=diff(log(GNP)), cond.dist='std'))
plot(gnp.g)   # for various graphics 

```



<br/> Example 5.6  

```r
library(TSA); library(xts)          # download and install if necessary
dENSO = detrend(ENSO, lowess=TRUE)
djiar = diff(log(djia$Close))[-1]

Keenan.test(dENSO)
Keenan.test(djiar)

Tsay.test(dENSO)
Tsay.test(djiar) 

test.linear(dENSO, main='ENSO')  
test.linear(djiar, main='DJIA Returns')  

```



<br/> Example 5.7

```r
library(xts)
djiar = diff(log(djia$Close))[-1]
acf2(djiar, col=3)     #  minimal autocorrelation  
acf2(djiar^2, col=4)   #  oozes autocorrelation 
library(fGarch)
summary(djia.g <- garchFit(~arma(1,0)+garch(1,1), data=djiar, cond.dist='std'))
plot(djia.g)    # to see all plot options

```

<br/> Example 5.8  

```r
library(xts)
djiar = diff(log(djia$Close))[-1]
library(fGarch)
summary(djia.ap <- garchFit(~arma(1,0)+aparch(1,1), data=djiar, cond.dist='std'))
plot(djia.ap)   # to see all plot options (none shown)

```



<br/> Example 5.9

```r
# Plot data with months as points
tsplot(flu, type='c')
points(flu, pch=Months, cex=1, col=2:5, font=2)
# Start analysis
dflu  = diff(flu)
lag1.plot(dflu, corr=FALSE)   # scatterplot with lowess fit  
thrsh = .05                   # threshold
Z = ts.intersect(dflu, lag(dflu,-1), lag(dflu,-2), lag(dflu,-3), lag(dflu,-4))
ind1  = ifelse(Z[,2] < thrsh, 1, NA)  # indicator < thrsh
ind2  = ifelse(Z[,2] < thrsh, NA, 1)  # indicator >= thrsh
X1    = Z[,1]*ind1
X2    = Z[,1]*ind2
summary(fit1 <- lm(X1~ Z[,2:5]) )         # case 1 
summary(fit2 <- lm(X2~ Z[,2:5]) )         # case 2
# Predictions
D     = cbind(rep(1, nrow(Z)), Z[,2:5])   # design matrix
p1    = D %*% coef(fit1)                 
p2    = D %*% coef(fit2)
prd   = ifelse(Z[,2] < thrsh, p1, p2)
tsplot(dflu, type='p', ylim=c(-.5,.5), pch=3, col=6, nym=2)
lines(prd, col=4, lwd=2)
 prde1 = sqrt(sum(resid(fit1)^2)/df.residual(fit1)) 
 prde2 = sqrt(sum(resid(fit2)^2)/df.residual(fit2))
 prde = ifelse(Z[,2] < thrsh, prde1, prde2)
    x = time(dflu)[-(1:4)]
   xx = c(x, rev(x))
   yy = c(prd - 2*prde, rev(prd + 2*prde))
polygon(xx, yy, border=8, col=gray(.6,   alpha=.2))
sarima(dflu-prd, 0,0,0)  # residual analysis (not shown)

library(NTS)       # load package - install it first
flutar = uTAR(diff(flu), p1=4, p2=4)   
sarima(resid(flutar), 0,0,0)  # residual analysis (not shown)

```

<br/> Example 5.10 & 5.11

```r
library(vars)
x = cbind(cmort, tempr, part)
summary( VAR(x, p=1, type='both') )     # 'both' fits constant + trend

VARselect(x, lag.max=10, type="both")
fit <- VAR(x, p=2, type="both") 
round(Bcoef(fit), 2)  # display all regression estimates
summary(fit)  # partial output  
acfm(resid(fit), 52)
serial.test(fit, lags.pt=12, type="PT.adjusted") 

( acfm(resid(fit), 0, plot=FALSE) )
(fit.pr = predict(fit, n.ahead = 24, ci = 0.95))  # 4 weeks ahead
fanchart(fit.pr)  # plot prediction + error bounds

```

<br/> Example 5.12

```r
library(marima)
model = define.model(kvar=3, ar=c(1,2), ma=c(1))
arp   = model$ar.pattern;  map = model$ma.pattern
resid(detr <- lm(cmort~time(cmort), na.action=NULL))
xdata = matrix(cbind(cmort.d, tempr, part), ncol=3)  # strip ts attributes
fit   = marima(xdata, ar.pattern=arp, ma.pattern=map, means=c(0,1,1), penalty=1)
#  resid analysis (not displayed)
innov = t(resid(fit));  plot.ts(innov);  acfm(innov, na.action=na.pass)
#  fitted values for cmort  
pred = ts(t(fitted(fit))[,1], start=start(cmort), freq=frequency(cmort))+detr$coef[1]+detr$coef[2]*time(cmort) 
tsplot(cmort, type='p', col=8, ylab="Cardiovascular Mortality")
lines(pred, col=4)
# print estimates and corresponding t^2-statistic
short.form(fit$ar.estimates, leading=FALSE) 
short.form(fit$ar.fvalues,   leading=FALSE)
# short.form(fit$ar.pvalues, leading=FALSE)   # p-values 
short.form(fit$ma.estimates, leading=FALSE)
short.form(fit$ma.fvalues,   leading=FALSE) 
# short.form(fit$ma.pvalues, leading=FALSE)   # p-values 
fit$resid.cov    # estimate of noise cov matrix  

```


[<sub>top</sub>](#table-of-contents)

---

## Chapter 6


<br/> Example 6.1

```r
tsplot(blood, type='o', col=c(4,6,3), pch=19, cex=1)

```

<br/> Example 6.2

```r
tsplot(cbind(gtemp_land, gtemp_both), col=astsa.col(c(4,6),.7), lwd=2, ylab='Temperature Deviations', spaghetti=TRUE)
legend("topleft", legend=c("Land Only","Land & Ocean"), col=c(4,6), lty=1, bty="n")

```

<br/> Example 6.5

```r
# generate data
set.seed(1)  
num = 50
w   = rnorm(num+1)
v   = rnorm(num)
mu  = cumsum(w)     # states:  mu[0], mu[1], . . ., mu[50] 
y   = mu[-1] + v    # obs:      y[1], . . ., y[50]
# filter and smooth (Ksmooth does both)  
ks = Ksmooth(y, A=1, mu0=0, Sigma0=1, Phi=1, sQ=1, sR=1)  

par(mfrow=c(3,1))
tsplot(mu[-1], type='p', col=4, pch=19, ylab=bquote(mu[~t]), main="Prediction", ylim=c(-3,8), gg=TRUE) 
 lines(ks$Xp, col=5, lwd=2)
 xx = c(1:50,50:1)
 yy = c(ks$Xp-2*sqrt(ks$Pp), rev(ks$Xp+2*sqrt(ks$Pp)))
 polygon(xx, yy, col=gray(.6,.2), border=NA)
 lines(y, col=6, lty=5)
tsplot(mu[-1], type='p', col=4, pch=19, ylab=bquote(mu[~t]), main="Filter", ylim=c(-3,8), gg=TRUE) 
 lines(ks$Xf, col=5, lwd=2)
 xx = c(1:50,50:1)
 yy = c(ks$Xf-2*sqrt(ks$Pf), rev(ks$Xf+2*sqrt(ks$Pf)))
 polygon(xx, yy, col=gray(.6,.2), border=NA)
 lines(y, col=6, lty=5)
tsplot(mu[-1], type='p', col=4, pch=19, ylab=bquote(mu[~t]), main="Smoother", ylim=c(-3,8), gg=TRUE) 
 lines(ks$Xs, col=5, lwd=2)
 xx = c(1:50,50:1)
 yy = c(ks$Xs-2*sqrt(ks$Ps), rev(ks$Xs+2*sqrt(ks$Ps)))
 polygon(xx, yy, col=gray(.6,.2), border=NA)
 lines(y, col=6, lty=5)

```


<br/> Example 6.6 

```r
# Generate Data
set.seed(90210);  num = 100
x = sarima.sim(n = num+1, ar = .8)
y = ts( x[-1] + rnorm(num) )

# Initial Estimates
u = ts.intersect(y, lag(y,-1), lag(y,-2))
varu = var(u); coru = cor(u)
phi = coru[1,3]/coru[1,2]
q = (1-phi^2)*varu[1,2]/phi;  r = varu[1,1] - q/(1-phi^2)
(init.par = c(phi, sqrt(q), sqrt(r)))  

# Function to evaluate the likelihood
Linn = function(para){
  phi = para[1]; sigw = para[2]; sigv = para[3]
  Sigma0 = (sigw^2)/(1-phi^2); Sigma0[Sigma0<0]=0
  kf = Kfilter(y, A=1, mu0=0, Sigma0, phi, sigw, sigv)
  return(kf$like)   }

# Estimation 
(est = optim(init.par, Linn, gr=NULL, method="BFGS", hessian=TRUE, control=list(trace=1, REPORT=1)))
SE = sqrt(diag(solve(est$hessian)))
round(cbind(estimate=c(phi=est$par[1],sigw=est$par[2],sigv=est$par[3]),SE), 3)

```


<br/> Example 6.7

```r
sl    = sd(window(gtemp_land, start=1991, end=2020))
sb    = sd(window(gtemp_both, start=1991, end=2020))
y     = cbind(gtemp_land/sl, gtemp_both/sb)
input = rep(1, nrow(y))
A     = matrix(c(1,1), nrow=2)
mu0 = -.35; Sigma0 = 1;  Phi = 1
# Function to Calculate Likelihood 
Linn=function(para){
 sQ = para[1]      # sigma_w
  sR1  = para[2]   # 11 element of sR
  sR2  = para[3]   # 22 element of sR
  sR21 = para[4]   # 21 element of sR
 sR = matrix(c(sR1,sR21,0,sR2), 2)  # put the matrix together
 drift = para[5]
 kf = Kfilter(y,A,mu0,Sigma0,Phi,sQ,sR,Ups=drift,Gam=NULL,input)  
 return(kf$like) 
 }
# Estimation
init.par = c(.1, 1, 1, 0, .05)  # initial values of parameters
(est = optim(init.par, Linn, NULL, method="BFGS", hessian=TRUE, control=list(trace=1,REPORT=1))) 
SE = sqrt(diag(solve(est$hessian))) 
# Summary of estimation  
estimate = est$par; u = cbind(estimate, SE)
rownames(u)=c("sigw","sR11", "sR22", "sR21", "drift"); u  
# Smooth (first set parameters to their final estimates)
sQ    = est$par[1]  
 sR1  = est$par[2]   
 sR2  = est$par[3]   
 sR21 = est$par[4]  
sR    = matrix(c(sR1,sR21,0,sR2), 2)
(R    = sR%*%t(sR))   #  to view the estimated R matrix
drift = est$par[5]  
ks    = Ksmooth(y,A,mu0,Sigma0,Phi,sQ,sR,Ups=drift,Gam=NULL,input)  
# Plot 
tsplot(y, spag=TRUE, type='o', pch=2:3, col=4:3, ylab='Temperature Deviations')
xsm  = ts(as.vector(ks$Xs), start=1850)
rmse = ts(sqrt(as.vector(ks$Ps)), start=1850)
lines(xsm, lwd=2, col=6)
  xx = c(time(xsm), rev(time(xsm)))
  yy = c(xsm-2*rmse, rev(xsm+2*rmse))
polygon(xx, yy, border=NA, col=gray(.6, alpha=.25))

```


<br/> Example 6.8

```r
library(nlme)   # loads package nlme (comes with R)
# Generate data (same as Example 6.6)
set.seed(999); num = 100; N = num+1
x = sarima.sim(ar=.8, n=N)
y = ts(x[-1] + rnorm(num))     
# Initial Estimates 
u = ts.intersect(y,lag(y,-1),lag(y,-2)) 
varu = var(u); coru = cor(u) 
phi = coru[1,3]/coru[1,2]             
q = (1-phi^2)*varu[1,2]/phi   
r = varu[1,1] - q/(1-phi^2) 
mu0 = 0; Sigma0 = 2.8
# run EM - note: input the variances q and r
( em = EM(y, A=1, mu0, Sigma0, phi, q, r) )   
# Standard Errors  (this uses nlme)
phi = em$Phi; sq = sqrt(em$Q); sr = sqrt(em$R)
mu0 = em$mu0; Sigma0 = em$Sigma0
para = c(phi, sq, sr)
# evaluate likelihood at estimates 
Linn=function(para){
  kf = Kfilter(y, A=1, mu0, Sigma0, para[1], para[2], para[3])
  return(kf$like) 
  }
emhess = fdHess(para, function(para) Linn(para))
SE = sqrt(diag(solve(emhess$Hessian)))  
# Display summary of estimation 
estimate = c(para, em$mu0, em$Sigma0); SE = c(SE,NA,NA)
u = cbind(estimate, SE)
rownames(u) = c("phi","sigw","sigv","mu0","Sigma0"); u

```

<br/> Example 6.9

```r
set.seed(1)
num = 100
phi1 = 1.5; phi2 =-.75   # the AR parameters
# simulate the AR(2) states [var(w[t]) = 1 by default] 
x = sarima.sim(ar = c(phi1, phi2), n=num)

# the observations
y = 50 + x + rnorm(num, 0, sqrt(.1)) # [var(v[t]) = .1]
# initial conditions (stationary values)
mux    = rbind(0, 0)
Sigmax = matrix(c(8.6,7.4,7.4,8.6), 2, 2)

# for estimation, we use these not so great starting values
Phi = diag(0, 2);  Phi[2,1] = 1; Phi[1,1] = .1; Phi[1,2] = .1
Q   = diag(0, 2);  Q[1,1]   = .1
R   = .1;          Gam = mean(y)

# run EM one at a time, then re-constrain the parameters
A = cbind(1, 0) 
input = rep(1, num)
for (i in 1:75){
 em = EM(y, A, mu0=mux, Sigma0=Sigmax, Phi, Q, R, Ups=NULL, Gam, input, max.iter=1)
 Phi = diag(0,2); Phi[2,1] = 1
 Phi[1,1] = em$Phi[1,1]; Phi[1,2] = em$Phi[1,2]
 Q = diag(0, 2); Q[1,1] = em$Q[1,1]; 
 R = em$R; Gam = em$Gam
}

Phi[1,1:2]   # (actual 1.5 and -.75)
Q[1,1]   # (actual 1)
R        # (actual .1)
Gam      # (actual 50)

```


<br/> Example 6.10

```r
y    = blood  # missing values are NA
num  = nrow(y)
A    = array(diag(1,3), dim=c(3,3,num))  # measurement matrices
for (k in 1:num) if (is.na(y[k,1])) A[,,k] = diag(0,3) 
# Initial values
mu0    = matrix(0,3,1)
Sigma0 = diag(c(.1,.1,1) ,3)
Phi    = diag(1, 3)
Q      = diag(c(.01,.01,1), 3) 
R      = diag(c(.01,.01,1), 3) 
# Run EM
(em = EM(y, A, mu0, Sigma0, Phi, Q, R)) 

# Run smoother at the estimates  
sQ = em$Q %^% .5  # for matrices, can use square root
sR = sqrt(em$R)
ks = Ksmooth(y, A, em$mu0, em$Sigma0, em$Phi, sQ, sR)

# Pull out the values
y1s = ks$Xs[1,,]
y2s = ks$Xs[2,,]
y3s = ks$Xs[3,,]
p1  = 2*sqrt(ks$Ps[1,1,])
p2  = 2*sqrt(ks$Ps[2,2,])
p3  = 2*sqrt(ks$Ps[3,3,])

# plots
miss = ifelse(is.na(y), 1 ,0)[,1]   # for ticks at missing days
par(mfrow=c(3,1))
tsplot(WBC, type='p', pch=19, ylim=c(1,5), col=6, lwd=2, cex=1)
 lines(y1s)
  xx = c(time(WBC), rev(time(WBC)))  # same for all
  yy = c(y1s-p1, rev(y1s+p1))
 polygon(xx, yy, border=8, col=astsa.col(8, alpha = .1))         
 lines(miss, type='h', lwd=2)
tsplot(PLT, type='p', ylim=c(3,6), pch=19, col=4, lwd=2, cex=1)
 lines(y2s)
  yy = c(y2s-p2, rev(y2s+p2))
 polygon(xx, yy, border=8, col=astsa.col(8, alpha = .1))       
 lines(3*miss, type='h', lwd=2)
tsplot(HCT, type='p', pch=19, ylim=c(20,40), col=2, lwd=2, cex=1)
 lines(y3s)
  yy = c(y3s-p3, rev(y3s+p3))
 polygon(xx, yy, border=8, col=astsa.col(8, alpha = .1))
 lines(20*miss, type='h', lwd=2)

```



<br/> Example 6.11

```r
A = cbind(1,1,0,0)  # measurement matrix  
# Function to Calculate Likelihood 
Linn = function(para){
 Phi = diag(0,4) 
 Phi[1,1] = para[1] 
 Phi[2,]=c(0,-1,-1,-1); Phi[3,]=c(0,1,0,0); Phi[4,]=c(0,0,1,0)
 sQ1 = para[2]; sQ2 = para[3]     # sqrt q11 and sqrt q22
 sQ  = diag(0,4); sQ[1,1]=sQ1; sQ[2,2]=sQ2
 sR = para[4]                     # sqrt r11
 kf = Kfilter(jj, A, mu0, Sigma0, Phi, sQ, sR)
 return(kf$like)  
 }

# Initial Parameters 
mu0      = c(.7,0,0,0) 
Sigma0   = diag(.04, 4)  
init.par = c(1.03, .1, .1, .5)   # Phi[1,1], the 2 sQs and sR

# Estimation
est = optim(init.par, Linn, NULL, method="BFGS", hessian=TRUE, control=list(trace=1,REPORT=1))
SE  = sqrt(diag(solve(est$hessian)))
u   = cbind(estimate=est$par, SE)
rownames(u)=c("Phi11","sigw1","sigw2","sigv"); u 
Phi      = diag(0,4) 
Phi[1,1] = est$par[1]; Phi[2,]  = c(0,-1,-1,-1) 
Phi[3,]  = c(0,1,0,0); Phi[4,]  = c(0,0,1,0)
sQ       = diag(0,4)
sQ[1,1]  = est$par[2]
sQ[2,2]  = est$par[3]   
sR       = est$par[4]   
ks       = Ksmooth(jj, A, mu0, Sigma0, Phi, sQ, sR)   

# Plots
Tsm   = ts(ks$Xs[1,,], start=1960, freq=4)
Ssm   = ts(ks$Xs[2,,], start=1960, freq=4)
p1    = 3*sqrt(ks$Ps[1,1,]); p2 = 3*sqrt(ks$Ps[2,2,])
par(mfrow=c(2,1))
tsplot(Tsm, main='Trend Component', ylab='', col=4)
  xx  = c(time(jj), rev(time(jj)))
  yy  = c(Tsm-p1, rev(Tsm+p1))
 polygon(xx, yy, border=NA, col=gray(.5, alpha=.3))
tsplot(Ssm, main='Seasonal Component', ylab='', col=4)
  xx  = c(time(jj), rev(time(jj)) )
  yy  = c(Ssm-p2, rev(Ssm+p2)) 
 polygon(xx, yy, border=NA, col=gray(.5, alpha=.3)) 

# Forecasts 
n.ahead = 12
num   = length(jj)
Xp    = ks$Xf[,,num]
Pp    = as.matrix(ks$Pf[,,num])
y     = c(jj[num])
rmspe = c(0)
for (m in 1:n.ahead){
  kf       = Kfilter(y[m], A, mu0=Xp, Sigma0=Pp, Phi, sQ, sR)
  Xp       = kf$Xp[,,1]
  Pp       = as.matrix(kf$Pp[,,1])
  sig      = A%*%Pp%*%t(A) + sR^2
  y[m]     = A%*%Xp
  rmspe[m] = sqrt(sig)
 }  
y = ts(append(jj, y), start=1960, freq=4)

# plot
dev.new()
tsplot(window(y, start=1975), type='o', main='', ylab='J&J QE/Share', col=4, ylim=c(5,26))
 lines(window(y, start=1981), type='o', col=6)
 upp = window(y, start=1981)+3*rmspe
 low = window(y, start=1981)-3*rmspe
  xx  = c(time(low), rev(time(upp)))
  yy  = c(low, rev(upp))
 polygon(xx, yy, border=NA, col=gray(.6, alpha = .2))

```



<br/> Example 6.13

```r
# Preliminary analysis
fit1   = sarima(cmort, 2,0,0, xreg=time(cmort))

acf(cbind(dmort <- resid(fit1$fit), tempr, part))
lag2.plot(tempr, dmort, 8)
lag2.plot(part, dmort, 8)

# easy prelim method: detrend cmort then do the regression
dcmort = detrend(cmort)
ded = ts.intersect(dM=dcmort, dM1=lag(dcmort,-1), dM2=lag(dcmort,-2),  T1=lag(tempr,-1), P=part, P4 = lag(part,-4), dframe=TRUE)
sarima(ded$dM, 0,0,0, xreg=ded[,2:6])  

##-- full run using Kfilter --## 
trend  =  time(cmort) - mean(time(cmort))   # center time
const  =  time(cmort)/time(cmort)           # a ts of 1s
ded = ts.intersect(M=cmort, T1=lag(tempr,-1), P=part, P4=lag(part,-4), trend, const)
y = ded[,1]; input =ded[,2:6]
A = matrix(c(1,0), 1,2)
# Function to Calculate Likelihood
Linn=function(para){
  phi1  = para[1]; phi2=para[2]; sR=para[3]; b1=para[4]; 
  b2    = para[5]; b3=para[6]; b4=para[7]; alf=para[8]
  mu0   = matrix(c(0,0), 2, 1); Sigma0 = diag(100, 2)
  Phi   = matrix(c(phi1, phi2, 1, 0), 2)
  sQ    = matrix(c(phi1, phi2), 2)*sR
   S    = 1
  Ups   = matrix(c(b1, 0, b2, 0, b3, 0, 0, 0, 0, 0), 2, 5)
  Gam   = matrix(c(0, 0, 0, b4, alf), 1, 5);   
  kf    = Kfilter(y, A, mu0, Sigma0, Phi, sQ, sR, Ups, Gam, input, S, version=2)
  return(kf$like) }
# Estimation
init.par = c(phi1=.3, phi2=.3, cR=5, b1=-.2, b2=.1, b3=.05, b4=-1.6, alf=mean(cmort)) 
L = c(.1, .1, 2, -.5,  0,  0, -2, 70)   # lower bound on parameters
U = c(.5, .5, 8,   0, .4, .2,  0, 90)   # upper bound - used in optim
est = optim(init.par, Linn, NULL, method="L-BFGS-B", lower=L, upper=U, hessian=TRUE, control=list(trace=1,REPORT=1,factr=10^8))
SE = sqrt(diag(solve(est$hessian)))
# Results
u = cbind(estimate=est$par, SE)
rownames(u)=c("phi1","phi2","sigv","TL1","P","PL4","trend",'constant')
round(u,3)

# Residual Analysis (not shown)
phi1   = est$par[1]; phi2 = est$par[2]
sR     = est$par[3]; b1   = est$par[4]
b2     = est$par[5]; b3   = est$par[6]
b4     = est$par[7]; alf  = est$par[8]
mu0    = matrix(c(0,0), 2, 1)
Sigma0 = diag(100, 2)
Phi    = matrix(c(phi1, phi2, 1, 0), 2)
S      = 1
Ups    = matrix(c(b1, 0, b2, 0, b3, 0, 0, 0, 0, 0), 2, 5)
Gam    = matrix(c(0, 0, 0, b4, alf), 1, 5) 
sQ     = matrix(c(phi1, phi2), 2)*sR
kf     = Kfilter(y, A, mu0, Sigma0, Phi, sQ, sR, Ups, Gam, input, S, version=2)
res    = ts(drop(kf$innov), start=start(cmort), freq=frequency(cmort))
sarima(res, 0,0,0, no.constant=TRUE)  # gives a full residual analysis

# complete ARMAX approach
ded = ts.intersect(M=cmort, M1=lag(cmort,-1), M2=lag(cmort,-2), T1=lag(tempr,-1), P=part, P4=lag(part,-4), trend=time(cmort), dframe=TRUE)
sarima(ded$M, 0,0,0, xreg=ded[,2:7])   

```



<br/> Example 6.14

```r
# data plot  
tsplot(cbind(qinfl, qintr), ylab='Rate (%)', col=c(4,6), spag=TRUE, type='o', pch=2:3)
legend("topleft", c("Inflation","Interest"), lty=1, col=c(4,6), pch=2:3, bg='white')
# set up 
y     = window(qinfl, c(1953,1), c(1965,2))  # quarterly inflation   
z     = window(qintr, c(1953,1), c(1965,2))  # interest   
num   = length(y) 
A     = array(z, dim=c(1,1,num))
input = matrix(1,num,1)  
# Function to Calculate Likelihood   
Linn  = function(para, y.data){  # pass data also
   phi = para[1];  alpha = para[2]
   b   = para[3];  Ups   = (1-phi)*b
   sQ  = para[4];  sR    = para[5]  
   kf  = Kfilter(y.data, A, mu0, Sigma0, phi, sQ, sR, Ups, Gam=alpha, input)  
   return(kf$like)    
}
# MLE   
mu0      =  1
Sigma0   = .01  
init.par = c(phi=.84, alpha=-.77, b=.85, sQ=.12, sR=1.1) # initial values   
est = optim(init.par,  Linn, NULL, y.data=y, method="BFGS", hessian=TRUE, 
             control=list(trace=1, REPORT=1, reltol=.0001))  
SE  = sqrt(diag(solve(est$hessian)))   
# results 
phi   = est$par[1];  alpha = est$par[2]
b     = est$par[3];  Ups   = (1-phi)*b         
sQ    = est$par[4];  sR    = est$par[5] 
round(cbind(estimate=est$par, SE), 3)  

# BEGIN BOOTSTRAP   
tol   = .0001     # determines convergence of optimizer     
nboot = 500       # number of bootstrap replicates 
# Run the filter at the estimates 
kf  = Kfilter(y, A, mu0, Sigma0, phi, sQ, sR, Ups, Gam=alpha, input) 
# Pull out necessary values from the filter and initialize  
xp      = kf$Xp
Pp      = kf$Pp
innov   = kf$innov 
sig     = kf$sig 
e       = innov/sqrt(sig)
e.star  = e                      # initialize values
y.star  = y  
xp.star = xp  
k       = 4:50                   # hold first 3 observations fixed 
para.star = matrix(0, nboot, 5)  # to store estimates
init.par  =  c(.84, -.77, .85, .12, 1.1)    
pb = txtProgressBar(min=0, max=nboot, initial=0, style=3)  # progress bar
for (i in 1:nboot){
 setTxtProgressBar(pb,i)                       
 e.star[k] = sample(e[k], replace=TRUE)   
 for (j in k){ 
   K = (phi*Pp[j]*z[j])/sig[j]  
  xp.star[j] = phi*xp.star[j-1] + Ups + K*sqrt(sig[j])*e.star[j]
  } 
   y.star[k] = z[k]*xp.star[k] + alpha + sqrt(sig[k])*e.star[k]  
 est.star  = optim(init.par, Linn, NULL, y.data=y.star, method='BFGS',control=list(reltol=tol))     
 para.star[i,] = cbind(est.star$par[1], est.star$par[2], est.star$par[3], abs(est.star$par[4]), abs(est.star$par[5]))   
}
close(pb) 

# SEs from the bootstrap (compare these to the SEs above)   
rmse = rep(NA,5)
for(i in 1:5){
  rmse[i]=sqrt(sum((para.star[,i]-est$par[i])^2)/nboot)
  cat(i, rmse[i],"\n") 
}           
# Plot phi v sigw 
phi  = para.star[,1] 
sigw = abs(para.star[,4]) 
phi  = ifelse(phi<0, NA, phi)    # any phi < 0 not plotted
scatter.hist(sigw, phi, ylab=bquote(phi), xlab=bquote(sigma[~w]), hist.col=astsa.col(5,.3), pt.col=astsa.col(5,.7), pt.size=1.2)
quantile(phi, na.rm=TRUE, c(.025, .5, .975))

```

<br/> Example 6.15

```r
set.seed(123)
num = 50
w = rnorm(num,0,.1)
x = cumsum(cumsum(w))  # states
y = x + rnorm(num,0,1) # observations
tsplot(cbind(x,y), ylab="", type='o', pch=c(NA,20), lwd=2:1, col=c(1,4), spag=TRUE, gg=TRUE)
# state space 
Phi = matrix(c(2,1,-1,0),2)
A   = matrix(c(1,0),1)
mu0 = matrix(0,2)
Sigma0 = diag(1,2)
Linn = function(para){
  sigw = para[1]; sigv = para[2]  
  sQ = diag(c(sigw,0))
  kf = Kfilter(y,A,mu0,Sigma0,Phi,sQ,sigv)
  return(kf$like)   
}
# estimation  
init.par=c(.1, 1)  
(est = optim(init.par, Linn, NULL, method="BFGS", hessian=TRUE, control=list(trace=1,REPORT=1))) 
SE = sqrt(diag(solve(est$hessian))) 
estimate = est$par; u = cbind(estimate, SE)
rownames(u)=c("sigw","sigv"); u 

# smooth
sigw = est$par[1]
sQ   = diag(c(sigw,0))
sigv = est$par[2]
ks = Ksmooth(y, A, mu0, Sigma0, Phi, sQ, sigv)
xsmoo = ts(ks$Xs[1,1,])
psmoo = ts(ks$Ps[1,1,])
upp   = xsmoo + 2*sqrt(psmoo)
low   = xsmoo - 2*sqrt(psmoo)
lines(xsmoo, col=6, lty=5, lwd=2)  
 xx = c(time(xsmoo), rev(time(xsmoo)))
 yy = c(low, rev(upp)) 
polygon(xx, yy, col=gray(.6,.2), border=NA)
lines(smooth.spline(y), lty=1, col=7)
legend("topleft", c("Observations","State"), pch=c(20,NA), lty=1, lwd=c(1,2), col=c(4,1), bty='n')
legend("bottomright", c("Smoother", "GCV Spline"), lty=c(5,1), lwd=c(2,1), col=c(6,7), bty='n')

```



<br/> Example 6.17

```r
library(depmixS4)
model <- depmix(EQcount ~1, nstates=2, data=data.frame(EQcount), family=poisson())
set.seed(90210)
fm <- fit(model)   # estimation 
summary(fm)
#-- get parameters --#
# make sure state 1 is min lambda 
u = as.vector(getpars(fm)) 
 if (u[7] <= u[8]) { para.mle = c(u[3:6], exp(u[7]), exp(u[8])) 
  } else {  para.mle = c(u[6:3], exp(u[8]), exp(u[7])) 
 }

( mtrans = matrix(para.mle[1:4], byrow=TRUE, nrow=2) )  
( lams = para.mle[5:6] )
( SE = standardError(fm)$se[7:8]*lams )  # see footnote
c( pi1 <- mtrans[2,1]/(2 - mtrans[1,1] - mtrans[2,2]), pi2 <- 1 - pi1 )
##-- Graphics --##
layout(matrix(c(1,2,1,3), 2))
tsplot(EQcount, type='c', ylim=c(4,42), col=8)
 states = ts(fm@posterior, start=1900)
 text(EQcount, col=6*states[,1]-2, labels=states[,1], cex=.9)
# prob of state 2
tsplot(states[,2], ylab=bquote(hat(pi)[~2]*' (t | n)'), col=4)
 abline(h=.5, col=6, lty=2)
# histogram
hist(EQcount, breaks=30, prob=TRUE, main=NA, col='lightblue')
 xvals = seq(1,45)
 u1 = pi1*dpois(xvals, lams[1])
 u2 = pi2*dpois(xvals, lams[2])
 lines(xvals, u1, col=4, lwd=2)
 lines(xvals, u2, col=2, lwd=2)

```

<br/> Example 6.18

```r
library(depmixS4)
y = ts(sp500w, start=2003, freq=52)  # makes data useable for depmix
mod3 <- depmix(y~1, nstates=3, data=data.frame(y))
set.seed(2)
# output (not displayed)
summary(fm3 <- fit(mod3))   # transition matrix and normal estimates
( SE = standardError(fm3) ) # corresponding SEs 
# graphics  
para.mle = as.vector(getpars(fm3)[-(1:3)])
# for display (states 1 and 3 names switched)
permu = matrix(c(0,0,1,0,1,0,1,0,0), 3,3) 
(mtrans.mle = permu%*%round(t(matrix(para.mle[1:9],3,3)),3)%*%permu)
(norms.mle =  round(matrix(para.mle[10:15],2,3),3)%*%permu)
layout(matrix(c(1,2, 1,3), 2), heights=c(1,.75))
tsplot(y, main=NA, ylab='S&P500 Weekly Returns', col=8, ylim=c(-.15,.11))
 culer = fm3@posterior[,1] 
 culer[culer==1]=4
 text(y, col=culer, labels=4-fm3@posterior[,1], cex=1.1)
acf1(ts(y^2), 20, col=4, xlab='LAG', main=NA, ylim=c(-.1,.3)) 
hist(y, 25, prob=TRUE, main="", xlab='S&P500 Weekly Returns', ylim=c(0,22), col=gray(.7,.2))
 Grid(minor=FALSE)
 culer=c(3,2,4); pi.hat = table(fm3@posterior[,1])/length(y) 
 for (i in 1:3) { mu=norms.mle[1,i]; sig = norms.mle[2,i]
  x = seq(-.15,.12, by=.001)
  lines(x, pi.hat[4-i]*dnorm(x, mean=mu, sd=sig), lwd=2, col=culer[i]) }

```



<br/> Example 6.19

```r
library(MSwM)  
dflu =  diff(flu)
model = lm(dflu~ 1)
mod = msmFit(model, k=2, p=2, sw=rep(TRUE,4))  # 2 regimes, AR(2)s 
summary(mod)
plotProb(mod, which=3)  # or which=2

```



<br/> Example 6.22

```r
y      = as.matrix(flu)
num    = length(y)
nstate = 4
M1     = as.matrix(cbind(1,0,1,0))  # normal
M2     = as.matrix(cbind(1,0,1,1))  # epi
prob   = matrix(0,num,1)  # to store pi2(t|t-1) 
yp     = y                # to store y(t|t-1)
xfilter = array(0, dim=c(nstate,1,num)) # to store x(t|t)
# Function to Calculate Likelihood 
Linn = function(para){
  alpha1= para[1]; alpha2= para[2]; beta= para[3]      
  sQ1= para[4];    sQ2= para[5];    sQ3= para[6] 
  sR =  para[7];   like= 0
  xf = matrix(0, nstate, 1)  # x filter
  xp = matrix(0, nstate, 1)  # x predict
  Pf = diag(.1, nstate)      # filter covar
  Pp = diag(.1, nstate)      # predict covar
  pi11 <- .75 -> pi22;  pi12 <- .25 -> pi21; pif1 <- .5 -> pif2            
  phi = diag(0, nstate)
  phi[1,1]= alpha1; phi[1,2]= alpha2; phi[2,1]= 1; phi[3,3]= 1 
  Ups = matrix(c(0,0,0,beta), nstate, 1)
  Q = diag(0, nstate)
  Q[1,1]= sQ1^2; Q[3,3]= sQ2^2; Q[4,4]= sQ3^2; R= sR^2
  # begin filtering 
    for(i in 1:num){
    xp = phi%*%xf + Ups; Pp = phi%*%Pf%*%t(phi) + Q
    sig1 = as.numeric(M1%*%Pp%*%t(M1) + R)
    sig2 = as.numeric(M2%*%Pp%*%t(M2) + R)
    k1 = Pp%*%t(M1)/sig1; k2 = Pp%*%t(M2)/sig2 
    e1 = y[i]-M1%*%xp; e2 = y[i]-M2%*%xp
    pip1 = pif1*pi11 + pif2*pi21
    pip2 = pif1*pi12 + pif2*pi22;  
    den1 = (1/sqrt(sig1))*exp(-.5*e1^2/sig1); 
    den2 = (1/sqrt(sig2))*exp(-.5*e2^2/sig2);
    denom = pip1*den1 + pip2*den2;
    pif1 = pip1*den1/denom;  pif2 = pip2*den2/denom;
    pif1 = as.numeric(pif1); pif2 = as.numeric(pif2)
    e1 = as.numeric(e1);     e2   = as.numeric(e2)
    xf = xp + pif1*k1*e1 + pif2*k2*e2
    eye = diag(1, nstate)
    Pf = pif1*(eye-k1%*%M1)%*%Pp + pif2*(eye-k2%*%M2)%*%Pp 
    like = like - log(pip1*den1 + pip2*den2)
    prob[i]<<-pip2; xfilter[,,i]<<-xf; innov.sig<<-c(sig1,sig2)
    yp[i]<<-ifelse(pip1 > pip2, M1%*%xp, M2%*%xp)  
    }    
 return(like)   
 }

# Estimation
alpha1=1.4; alpha2=-.5; beta=.3; sQ1=.1; sQ2=.1; sQ3=.1;  sR=.1
init.par = c( alpha1, alpha2, beta, sQ1, sQ2, sQ3, sR)
(est = optim(init.par, Linn, NULL, method="BFGS", hessian=TRUE, control=list(trace=1,REPORT=1)))
SE   = sqrt(diag(solve(est$hessian)))    
u    = cbind(estimate=est$par, SE)
rownames(u) = c("alpha1","alpha2","beta","sQ1","sQ2","sQ3",'sR')
round(u, 3)

# Graphics 
predepi =  ifelse(prob<.5,1,2)  
k = 6:length(y)      
Time = time(flu)[k]
regime = predepi[k]
culer = ifelse(regime==1,4,2)
par(mfrow=2:1 )
tsplot(Time, y[k], col=8)
 text(Time, y[k], col=culer, labels=regime, cex=1.1)  
 text(1979,.8,"(a)") 
tsplot(Time, xfilter[1,,k], ylim=c(-.1,.4), ylab="", col=4)
 lines(Time, xfilter[3,,k], col=3); 
 lines(Time, xfilter[4,,k], col=2)
 text(1979,.38,"(b)")

```




<br/> Example 6.24

```r
# generate states and obs
set.seed(1)
sQ = 1; sR = 3; n = 100
mu0 = 0; Sigma0 = 10; x0 = rnorm(1, mu0, Sigma0)
w  = rnorm(n);  v = rnorm(n)
x = c(x0   + sQ*w[1])  # initialize states
y = c(x[1] + sR*v[1])  # initialize obs
for (t in 2:n){ 
  x[t] = x[t-1] + sQ*w[t]
  y[t] = x[t]   + sR*v[t]   }
# set up the Gibbs sampler
burn   = 50;  n.iter = 1000
niter  = burn + n.iter
draws  = c()
# priors for R (a,b) and Q (c,d) IG distributions
a = 2; b = 2; c = 2; d = 1  
# (1) initialize - sample sR and sQ  
sR = sqrt(1/rgamma(1,a,b));  sQ = sqrt(1/rgamma(1,c,d))
# progress bar
pb = txtProgressBar(min=0, max=niter, initial=0, style=3)  
# run it
for (iter in 1:niter){
setTxtProgressBar(pb, iter)
# sample the states  
 run   = ffbs(y, A=1, mu0=0, Sigma0=10, Phi=1, sQ, sR) 
# sample the parameters    
 xs    = as.matrix(run$Xs)
  R    = 1/rgamma(1, a+n/2, b+sum((y-xs)^2)/2)
 sR    = sqrt(R)
  Q    = 1/rgamma(1, c+(n-1)/2, d+sum(diff(xs)^2)/2)
 sQ    = sqrt(Q)
# store everything 
draws = rbind(draws, c(sQ,sR,xs))   }
close(pb)
# pull out the results for plotting
draws  = draws[(burn+1):(niter),]
 q025  = function(x){quantile(x, 0.025)}
 q975  = function(x){quantile(x, 0.975)}
xs     = draws[, 3:(n+2)]
lx     = apply(xs, 2, q025)
mx     = apply(xs, 2, mean)
ux     = apply(xs, 2, q975)
# plot states, data, and smoother distn
tsplot(cbind(x,y,mx), spag=TRUE, lwd=c(1,1,2), ylab='', col=c(7,5,6), type='o', pch=c(NA,20,NA), gg=TRUE)
a = bquote(X[~t]); b = bquote(Y[~t]); c = bquote(X[~t]^n)
legend('topleft', legend=c(a,b,c), lty=1, lwd=c(1,1,2), col=c(7,5,6), bty="n", pch=c(NA,20,NA))
 xx=c(1:100, 100:1)
 yy=c(lx, rev(ux))
polygon(xx, yy, border=8, col=gray(.7,.2)) 
# plot parameters
scatter.hist(draws[,1],draws[,2], xlab=bquote(sigma[w]), ylab=bquote(sigma[v]), reset.par = FALSE, pt.col=5, hist.col=5)
abline(v=mean(draws[,1]), col=3, lwd=2)
abline(h=mean(draws[,2]), col=3, lwd=2)

```




<br/> Example 6.25

```r
 set.seed(90013)     # Skid Row
 x = sarima.sim(ar=c(1,-.9)) + 50  # phi0 = 50(1-1+.9) = 45
 ar.mcmc(x, 2)  

```



<br/> Example 6.26 

```r
set.seed(90210)
n   = length(jj)
A   = matrix(c(1,1,0,0), 1, 4)
Phi = diag(0,4)
  Phi[1,1] = 1.03 
  Phi[2,]  = c(0,-1,-1,-1); Phi[3,]=c(0,1,0,0); Phi[4,]=c(0,0,1,0)
mu0 = rbind(.7,0,0,0)
Sigma0 = diag(.04, 4)
sR = 1                    # observation noise standard deviation
sQ = diag(c(.1,.1,0,0))   # state noise standard deviations on the diagonal
# initializing and hyperparameters
burn   = 50
n.iter = 1000
niter  = burn + n.iter
draws  = NULL
a = 2; b = 2; c = 2; d = 1   # hypers (c and d for both Qs)
pb = txtProgressBar(min = 0, max = niter, initial = 0, style=3)  # progress bar
# start Gibbs
for (iter in 1:niter){
# draw states 
  run  = ffbs(jj,A,mu0,Sigma0,Phi,sQ,sR)   # initial values are given above
  xs   = run$Xs
# obs variance
  R    = 1/rgamma(1,a+n/2,b+sum((as.vector(jj)-as.vector(A%*%xs[,,]))^2))
 sR    = sqrt(R)
# beta where phi = 1+beta  
  Y    = diff(xs[1,,])
  D    = as.vector(lag(xs[1,,],-1))[-1]
 regu  = lm(Y~0+D)  # est beta = phi-1
 phies = as.vector(coef(summary(regu)))[1:2] + c(1,0) # phi estimate and SE
 dft   = df.residual(regu)
 Phi[1,1]  = phies[1] + rt(1,dft)*phies[2]  # use a t to sample phi
# state variances
  u   = xs[,,2:n] - Phi%*%xs[,,1:(n-1)]
  uu  = u%*%t(u)/(n-2)
  Q1  = 1/rgamma(1,c+(n-1)/2,d+uu[1,1]/2)
  sQ1 = sqrt(Q1)
  Q2  = 1/rgamma(1,c+(n-1)/2,d+uu[2,2]/2)
  sQ2 = sqrt(Q2) 
  sQ  = diag(c(sQ1, sQ2, 0,0))
# store results
 trend = xs[1,,]
 season= xs[2,,] 
 draws = rbind(draws,c(Phi[1,1],sQ1,sQ2,sR,trend,season))
 setTxtProgressBar(pb,iter)  
}
close(pb)

##-- graphics --##
 u     = draws[(burn+1):(niter),]
 parms = u[,1:4]
 q025  = function(x){quantile(x,0.025)}
 q975  = function(x){quantile(x,0.975)}

#  plot parameters  
 names= c(bquote(phi), bquote(sigma[w1]), bquote(sigma[w2]), bquote(sigma[v]))
par(mfrow=c(2,2))
for (i in 1:4){
 hist(parms[,i], col=astsa.col(5,.4), main=names[i], xlab='')
 u1 = apply(parms,2,q025); u2 = apply(parms,2,mean); u3 = apply(parms,2,q975);
 abline(v=c(u1[i], u2[i], u3[i]), lwd=2, col=c(3,6,3))
}

#  plot states   
dev.new()
  tr   = ts(u[,5:(n+4)], start=1960, frequency=4)
 ltr   = ts(apply(tr,2,q025), start=1960, frequency=4)
 mtr   = ts(apply(tr,2,mean), start=1960, frequency=4)
 utr   = ts(apply(tr,2,q975), start=1960, frequency=4)
par(mfrow=2:1)
tsplot(mtr, ylab='', col=4, main='trend', cex.main=1)
 xx = c(time(mtr), rev(time(mtr)))
 yy = c(ltr, rev(utr))
polygon(xx, yy, border=NA, col=astsa.col(4,.1)) 
#  season
  sea    = ts(u[,(n+5):(2*n)], start=1960, frequency=4)
 lsea    = ts(apply(sea,2,q025), start=1960, frequency=4)
 msea    = ts(apply(sea,2,mean), start=1960, frequency=4)
 usea    = ts(apply(sea,2,q975), start=1960, frequency=4)
tsplot(msea, ylab='', col=4, main='season', cex.main=1)
 xx = c(time(msea), rev(time(msea)))
 yy = c(lsea, rev(usea))
polygon(xx, yy, border=NA, col=astsa.col(4,.1)) 

```




<br/> Example 6.31

```r
set.seed(90210)
x1 = rnorm(500)         # independent sampling
x2 = sarima.sim(ar=.5)  # good sampling
x3 = sarima.sim(ar=.99) # not so good sampling
round( apply(cbind(x1,x2,x3), 2, ESS) )

```

<br/> Example 6.32 

```r
spfit = SV.mcmc(sp500w)
str(spfit)  # use ?SV.mcmc for option descriptions

```

<br/> Example 6.33

```r
SV.mle(BCJ[,'boa'])   # also produces the graphics

```

<br/> Example 6.34

```r
SV.mle(BCJ[,"boa"], rho=0, feedback=TRUE)
SV.mle(BCJ[,"boa"], feedback=TRUE)

```



[<sub>top</sub>](#table-of-contents)

---

## Chapter 7

Code in Introduction

```r
x = matrix(0, 128, 6)
for (i in 1:6) x[,i] = rowMeans(fmri[[i]])
colnames(x) = rep(c("Brush", "Heat", "Shock"), 2) 
tsplot(x, ncol=2, byrow=FALSE, col=4:2, main=NA, ylim=c(-.6,.6))
mtext("Awake",   side=3, line=-1, adj=.25, cex=1, outer=TRUE)
mtext("Sedated", side=3, line=-1, adj=.78, cex=1, outer=TRUE)

```

```r
P = 1:1024; S = P+1024
x = eqexp[P, c(5:6,5:6+8,17)]
x = cbind(x, eqexp[S, c(5:6,5:6+8,17)])
tsplot(x, ncol=2, byrow=FALSE, col=2:6)
mtext("P waves", side=3, line=-1, adj=.25, cex=.9, outer=TRUE)
mtext("S waves", side=3, line=-1, adj=.78, cex=.9, outer=TRUE)

```


<br/> Example 7.1

```r
tsplot(climhyd, ncol=2, col=2:7)    # Figure 7.3
Y     = climhyd     # Y to hold the transformed series
Y[,6] = log(Y[,6])  # log inflow
Y[,5] = sqrt(Y[,5]) # sqrt precipitation
L = 25; M = 100; alpha = .001;  fdr = .001
nq = 2              # number of inputs  (Temp and Precip)
# Spectral Matrix
Yspec = mvspec(Y, spans=L, kernel="daniell", taper=.1, plot=FALSE)
 n = Yspec$n.used          # effective sample size
 Fr = Yspec$freq           # fundamental freqs 
 n.freq = length(Fr)       # number of frequencies
 Yspec$bandwidth           # = 0.05  
# Coherencies  
Fq = qf(1-alpha, 2, L-2)
cn = Fq/(L-1+Fq)
plt.name=c("(a)","(b)","(c)","(d)","(e)","(f)")
par(mfrow=c(2,3)) 
# The coherencies are listed as 1,2,..., 15=choose(6,2) 
for (i in 11:15){
 tsplot(Fr,Yspec$coh[,i], col=4, ylab="Coherence", xlab="Frequency", ylim=c(0,1), main=c("Inflow with", names(climhyd[i-10])), topper=1.5)
abline(h = cn); text(.45,.98, plt.name[i-10], cex=1.2)  } 
# Multiple Coherency 
coh.15 = stoch.reg(Y, cols.full = c(1,5), cols.red = NULL, alpha, L, M, plot.which = "NULL")  
tsplot(Fr,coh.15$coh, col=4, ylab="Coherence", xlab="Frequency",  ylim=c(0,1), topper=1.5)
abline(h = cn); text(.45,.98, plt.name[6], cex=1.2) 
title(main = c("Inflow with", "Temp and Precip"))
# Partial F (called eF; avoid use of F alone)
numer.df = 2*nq
denom.df = Yspec$df-2*nq
out.15 = stoch.reg(Y, cols.full=c(1,5), cols.red=5, alpha, L, M, plot.which = "F.stat")
layout(matrix(c(1,2,1,3), 2)) 
tsplot(Fr, out.15$eF, col=4, ylab="F", xlab="Frequency", main = "Partial F Statistic")
eF = out.15$eF
pvals = pf(eF, numer.df, denom.df, lower.tail = FALSE)
pID = FDR(pvals, fdr);  abline(h=c(eF[pID]), lty=2)
abline(h=qf(.001, numer.df, denom.df, lower.tail = FALSE) )
# Regression Coefficients
S = seq(from = -M/2+1, to = M/2 - 1, length = M-1)
tsplot(S, coh.15$Betahat[,1], type="h", xlab="Index", xlim=c(-20,20), main=names(climhyd[1]), ylim=c(-.03, .06), col=4, lwd=2, ylab="Impulse Response")
abline(h=0)
tsplot(S, coh.15$Betahat[,2], type="h", xlab="Index", xlim=c(-20,20), main=names(climhyd[5]), ylim=c(-.03, .06), col=4, lwd=2, ylab="Impulse Response")
abline(h=0)

```


<br/> Example 7.2

```r
attach(beamd)     # see warning in ?attach
tau = rep(0,3)
u = ccf(sensor1, sensor2, plot=FALSE)
tau[1] = u$lag[which.max(u$acf)]    #  17
u = ccf(sensor3, sensor2, plot=FALSE)
tau[3] = u$lag[which.max(u$acf)]    # -22
Y = ts.union(lag(sensor1,tau[1]), lag(sensor2, tau[2]), lag(sensor3, tau[3]))
Y = ts.union(Y, rowMeans(Y))
colnames(Y) = c(names(beamd), 'beamd')
tsplot(Y, col=4, main="Infrasonic Signals and Beam")
detach(beamd)     # Redemption

```

<br/> Example 7.4

```r
L     = 9; fdr = .001; N = 3
Y     = cbind(beamd, beam=rowMeans(beamd) )
n     = nextn(nrow(Y))
Y.fft = mvfft(as.ts(Y))/sqrt(n)
Df    = Y.fft[,1:3]  # fft of the data
Bf    = Y.fft[,4]    # beam fft
ssr   = N*Re(Bf*Conj(Bf))               # raw signal spectrum
sse   = Re(rowSums(Df*Conj(Df))) - ssr  # raw error spectrum
# Smooth
SSE   = filter(sse, sides=2, filter=rep(1/L,L), circular=TRUE)
SSR   = filter(ssr, sides=2, filter=rep(1/L,L), circular=TRUE)
SST   = SSE + SSR
par(mfrow=2:1) 
Fr    = 1:(n-1)/n
nFr   = 1:200     # number of freqs to plot
tsplot(Fr[nFr], log(SST[nFr]), ylab="log Power", col=5, xlab="", main="Sum of Squares")
lines(Fr[nFr], log(SSE[nFr]), col=6, lty=5)
eF  = (N-1)*SSR/SSE
df1 = 2*L
df2 = 2*L*(N-1)
# Compute F-value for false discovery probability of fdr
p   = pf(eF, df1, df2, lower=FALSE)
pID = FDR(p,fdr)
Fq  = qf(1-fdr, df1, df2)
tsplot(Fr[nFr], eF[nFr], col=5, ylab="F-statistic", xlab="Frequency", main="F Statistic", cex.main=1)
abline(h=c(Fq, eF[pID]), lty=c(1,5), col=8)

```


<br/> Example 7.6

```r
n         = 128               # length of series
n.freq    = 1 + n/2           # number of frequencies
Fr        = (0:(n.freq-1))/n  # the frequencies
N         = c(5,4,5,3,5,4)    # number of series for each cell
n.subject = sum(N)            # number of subjects (26)
n.trt     = 6                 # number of treatments
L         = 3                 # for smoothing
num.df    = 2*L*(n.trt-1)     # df for F test
den.df    = 2*L*(n.subject-n.trt)
# Design Matrix (Z):
Z1   = outer(rep(1,N[1]), c(1,1,0,0,0,0))
Z2   = outer(rep(1,N[2]), c(1,0,1,0,0,0))
Z3   = outer(rep(1,N[3]), c(1,0,0,1,0,0))
Z4   = outer(rep(1,N[4]), c(1,0,0,0,1,0))
Z5   = outer(rep(1,N[5]), c(1,0,0,0,0,1))
Z6   = outer(rep(1,N[6]), c(1,-1,-1,-1,-1,-1))
Z    = rbind(Z1, Z2, Z3, Z4, Z5, Z6)
ZZ   = t(Z)%*%Z
SSEF <- rep(NA, n) -> SSER
HatF = Z%*%solve(ZZ, t(Z))
HatR = Z[,1]%*%t(Z[,1])/ZZ[1,1]
par(mfrow=c(3,3), mar=c(3.5,4,0,0), oma=c(0,0,2,2), mgp = c(1.6,.6,0))
loc.name = c("Cortex 1","Cortex 2","Cortex 3","Cortex 4","Caudate","Thalamus 1","Thalamus 2","Cerebellum 1","Cerebellum 2")
for(Loc in 1:9) {
 i = n.trt*(Loc-1)
 Y = cbind(fmri[[i+1]], fmri[[i+2]], fmri[[i+3]], fmri[[i+4]], fmri[[i+5]], fmri[[i+6]])
 Y = mvfft(spec.taper(Y, p=.5))/sqrt(n)	
 Y = t(Y)       # Y is now 26 x 128 FFTs
# Calculation of Error Spectra
for (k in 1:n) {
  SSY    = Re(Conj(t(Y[,k]))%*%Y[,k])
  SSReg  = Re(Conj(t(Y[,k]))%*%HatF%*%Y[,k])
 SSEF[k] = SSY - SSReg
  SSReg  = Re(Conj(t(Y[,k]))%*%HatR%*%Y[,k])
 SSER[k] = SSY - SSReg  }
# Smooth
sSSEF    = filter(SSEF, rep(1/L, L), circular = TRUE)
sSSER    = filter(SSER, rep(1/L, L), circular = TRUE)
eF       = (den.df/num.df)*(sSSER-sSSEF)/sSSEF
tsplot(Fr, eF[1:n.freq], col=5, xlab="Frequency", ylab="F Statistic", ylim=c(0,7))
abline(h=qf(.999, num.df, den.df),lty=2)
text(.25, 6.5, loc.name[Loc], cex=1.2)   
}

```



<br/> Example 7.7

```r
n          = 128               # length of series
n.freq     = 1 + n/2           # number of frequencies
Fr         = (0:(n.freq-1))/n  # the frequencies 
N          = c(5,4,5,3,5,4)    # number of series for each cell
n.subject  = sum(N)            # number of subjects (26)
n.trt      = 6                 # number of treatments
L          = 3                 # for smoothing
num.df     = 2*L*(n.trt-1)     # dfs for F test
den.df     = 2*L*(n.subject-n.trt)


# Design Matrix (Z): 
Z1 = outer(rep(1,N[1]), c(1,1,0,0,0,0))
Z2 = outer(rep(1,N[2]), c(1,0,1,0,0,0))
Z3 = outer(rep(1,N[3]), c(1,0,0,1,0,0))
Z4 = outer(rep(1,N[4]), c(1,0,0,0,1,0)) 
Z5 = outer(rep(1,N[5]), c(1,0,0,0,0,1)) 
Z6 = outer(rep(1,N[6]), c(1,-1,-1,-1,-1,-1)) 

Z  = rbind(Z1, Z2, Z3, Z4, Z5, Z6)
ZZ = t(Z)%*%Z 

SSEF <- rep(NA, n) -> SSER   

HatF = Z%*%solve(ZZ, t(Z))
HatR = Z[,1]%*%t(Z[,1])/ZZ[1,1]

par(mfrow=c(3,3))
loc.name = c("Cortex 1","Cortex 2","Cortex 3","Cortex 4","Caudate","Thalamus 1",
              "Thalamus 2", "Cerebellum 1","Cerebellum 2")

for(Loc in 1:9) {   
 i = n.trt*(Loc-1)   
 Y = cbind(fmri[[i+1]], fmri[[i+2]], fmri[[i+3]], fmri[[i+4]], fmri[[i+5]], fmri[[i+6]])
 Y = mvfft(spec.taper(Y, p=.5))/sqrt(n)	
 Y = t(Y)      # Y is now 26 x 128 FFTs

 # Calculation of Error Spectra 
 for (k in 1:n) {   
  SSY = Re(Conj(t(Y[,k]))%*%Y[,k])
  SSReg = Re(Conj(t(Y[,k]))%*%HatF%*%Y[,k])
 SSEF[k] = SSY - SSReg
  SSReg = Re(Conj(t(Y[,k]))%*%HatR%*%Y[,k])
 SSER[k] = SSY - SSReg  
 }

# Smooth 
sSSEF = filter(SSEF, rep(1/L, L), circular = TRUE)
sSSER = filter(SSER, rep(1/L, L), circular = TRUE)

eF =(den.df/num.df)*(sSSER-sSSEF)/sSSEF

tsplot(Fr, eF[1:n.freq], xlab="Frequency", ylab="F Statistic", ylim=c(0,7), main=loc.name[Loc])
abline(h=qf(.999, num.df, den.df),lty=2) 
}

```

<br/> Example 7.7

```r
n         = 128 
n.freq    = 1 + n/2
Fr        = (0:(n.freq-1))/n  
nFr       = 1:(n.freq/2)
N         = c(5,4,5,3,5,4)   # number of subjects per cell
n.subject = sum(N)
n.para    = 6                # number of parameters
L         = 3                # for smoothing
df.stm    = 2*L*(3-1)        # stimulus (3 levels: Brush, Heat, Shock)
df.con    = 2*L*(2-1)        # conscious (2 levels: Awake, Sedated)
df.int    = 2*L*(3-1)*(2-1)  # interaction
den.df    = 2*L*(n.subject-n.para) # df for full model
# Design Matrix:           mu  a1  a2   b  g1  g2
 Z1  = outer(rep(1,N[1]), c(1,  1,  0,  1,  1,  0))
 Z2  = outer(rep(1,N[2]), c(1,  0,  1,  1,  0,  1))
 Z3  = outer(rep(1,N[3]), c(1, -1, -1,  1, -1, -1))
 Z4  = outer(rep(1,N[4]), c(1,  1,  0, -1, -1,  0))
 Z5  = outer(rep(1,N[5]), c(1,  0,  1, -1,  0, -1))
 Z6  = outer(rep(1,N[6]), c(1, -1, -1, -1,  1,  1))
Z    = rbind(Z1, Z2, Z3, Z4, Z5, Z6)
ZZ   = t(Z)%*%Z
c() -> SSEF-> SSE.stm -> SSE.con -> SSE.int
HatF    = Z%*%solve(ZZ,t(Z))
Hat.stm = Z[,-(2:3)]%*%solve(ZZ[-(2:3),-(2:3)], t(Z[,-(2:3)]))
Hat.con = Z[,-4]%*%solve(ZZ[-4,-4], t(Z[,-4]))
Hat.int = Z[,-(5:6)]%*%solve(ZZ[-(5:6),-(5:6)], t(Z[,-(5:6)]))
par(mfrow=c(5,3))
loc.name = c("Cortex 1","Cortex 2","Cortex 3","Cortex 4","Caudate", "Thalamus 1","Thalamus 2","Cerebellum 1","Cerebellum 2")
for(Loc in c(1:4,9)) {   # only Loc 1 to 4 and 9 used
 i = 6*(Loc-1)
 Y = cbind(fmri[[i+1]], fmri[[i+2]], fmri[[i+3]], fmri[[i+4]], fmri[[i+5]], fmri[[i+6]])
 Y = mvfft(spec.taper(Y, p=.5))/sqrt(n);  Y = t(Y)
for (k in 1:n) {
   SSY      = Re(Conj(t(Y[,k]))%*%Y[,k])
   SSReg    = Re(Conj(t(Y[,k]))%*%HatF%*%Y[,k])
 SSEF[k]    = SSY - SSReg
   SSReg    = Re(Conj(t(Y[,k]))%*%Hat.stm%*%Y[,k])
 SSE.stm[k] = SSY-SSReg
   SSReg    = Re(Conj(t(Y[,k]))%*%Hat.con%*%Y[,k])
 SSE.con[k] = SSY-SSReg
   SSReg    = Re(Conj(t(Y[,k]))%*%Hat.int%*%Y[,k])
 SSE.int[k] = SSY-SSReg    }
# Smooth
sSSEF    = filter(SSEF, rep(1/L, L), circular = TRUE)
sSSE.stm = filter(SSE.stm, rep(1/L, L), circular = TRUE)
sSSE.con = filter(SSE.con, rep(1/L, L), circular = TRUE)
sSSE.int = filter(SSE.int, rep(1/L, L), circular = TRUE)
eF.stm   = (den.df/df.stm)*(sSSE.stm-sSSEF)/sSSEF
eF.con   = (den.df/df.con)*(sSSE.con-sSSEF)/sSSEF
eF.int   = (den.df/df.int)*(sSSE.int-sSSEF)/sSSEF
tsplot(Fr[nFr], eF.stm[nFr], col=5, xlab="Frequency", ylab='F-Statistic', ylim=c(0,12), topper=.2, margins=c(0,1.75,0,0))
  abline(h=qf(.999, df.stm, den.df),lty=5, col=8)       
  if(Loc==1) mtext("Stimulus", side=3, line=0, cex=.9)
  mtext(loc.name[Loc], side=2, line=3, cex=.9)
tsplot(Fr[nFr], eF.con[nFr], col=5, xlab="Frequency", ylab='F-Statistic', ylim=c(0,12), topper=.2, margins=c(0,1,0,0))
  abline(h=qf(.999, df.con, den.df),lty=5, col=8)
  if(Loc==1)  mtext("Consciousness", side=3, line=0, cex=.9)   
tsplot(Fr[nFr], eF.int[nFr], col=5, xlab="Frequency", ylab='F-Statistic', ylim=c(0,12), topper=.2, margins=c(0,1,0,.2))
  abline(h=qf(.999, df.int, den.df), lty=5, col=8)
  if(Loc==1) mtext("Interaction", side=3, line=0, cex=.9)    
}

```



<br/> Example 7.8

```r
n  = 128; n.freq = 1 + n/2
Fr = (0:(n.freq-1))/n; nFr = 1:(n.freq/2)
N  = c(5,4,5,3,5,4); n.subject = sum(N); L = 3
# Design Matrix
Z1 = outer(rep(1,N[1]), c(1,0,0,0,0,0))
Z2 = outer(rep(1,N[2]), c(0,1,0,0,0,0))
Z3 = outer(rep(1,N[3]), c(0,0,1,0,0,0))
Z4 = outer(rep(1,N[4]), c(0,0,0,1,0,0))
Z5 = outer(rep(1,N[5]), c(0,0,0,0,1,0))
Z6 = outer(rep(1,N[6]), c(0,0,0,0,0,1))
Z  = rbind(Z1, Z2, Z3, Z4, Z5, Z6);  ZZ = t(Z)%*%Z
# Contrasts:  6 by 3
A  = rbind(diag(1,3), diag(1,3))
nq = nrow(A);  num.df = 2*L*nq; den.df = 2*L*(n.subject-nq)
HatF = Z%*%solve(ZZ, t(Z))   # full model
rep(NA, n) -> SSEF -> SSER; eF = matrix(0,n,3)
par(mfrow=c(5,3))
loc.name = c("Cortex 1", "Cortex 2", "Cortex 3", "Cortex 4", "Caudate", "Thalamus 1", "Thalamus 2", "Cerebellum 1", "Cerebellum 2")
cond.name = c("Brush", "Heat", "Shock")
for(Loc in c(1:4,9)) {
 i = 6*(Loc-1)
 Y = cbind(fmri[[i+1]], fmri[[i+2]], fmri[[i+3]], fmri[[i+4]], fmri[[i+5]], fmri[[i+6]])
 Y = mvfft(spec.taper(Y, p=.5))/sqrt(n); Y = t(Y)
 for (cond in 1:3){
  Q = t(A[,cond])%*%solve(ZZ, A[,cond])
  HR = A[,cond]%*%solve(ZZ, t(Z))
  for (k in 1:n){
    SSY    = Re(Conj(t(Y[,k]))%*%Y[,k])
    SSReg  = Re(Conj(t(Y[,k]))%*%HatF%*%Y[,k])
   SSEF[k] = (SSY-SSReg)*Q
    SSReg  = HR%*%Y[,k]
   SSER[k] = Re(SSReg*Conj(SSReg))  }
# Smooth
sSSEF  = filter(SSEF, rep(1/L, L), circular = TRUE)
sSSER  = filter(SSER, rep(1/L, L), circular = TRUE)
eF[,cond] = (den.df/num.df)*(sSSER/sSSEF)   }
tsplot(Fr[nFr], eF[nFr,1], col=5, xlab="Frequency", ylab="F Statistic", ylim=c(0,5), topper=.2, margins=c(0,1.75,0,0))
  abline(h=qf(.999, num.df, den.df),lty=5, col=8)       
  if(Loc==1) mtext("Brush", side=3, line=0, cex=.9)
  mtext(loc.name[Loc], side=2, line=3, cex=.9)
tsplot(Fr[nFr], eF[nFr,2], col=5, xlab="Frequency", ylab="F Statistic", ylim=c(0,5), topper=.2, margins=c(0,1,0,0))
  abline(h=qf(.999, num.df, den.df),lty=5, col=8)
  if(Loc==1)  mtext("Heat", side=3, line=0, cex=.9)   
tsplot(Fr[nFr],eF[nFr,3],  col=5,, xlab="Frequency", ylab="F Statistic", ylim=c(0,5), topper=.2, margins=c(0,1,0,.2))
  abline(h=qf(.999, num.df, den.df),lty=5, col=8)
  if(Loc==1) mtext("Shock", side=3, line=0, cex=.9)  
}  

```



<br/> Example 7.9

```r
P = 1:1024; S = P+1024; N = 8; n = 1024; p.dim = 2; m = 10; L = 2*m+1
eq.P   = as.ts(eqexp[P,1:8]);  eq.S = as.ts(eqexp[S,1:8])
eq.m   = cbind(rowMeans(eq.P), rowMeans(eq.S))
ex.P   = as.ts(eqexp[P,9:16]);  ex.S = as.ts(eqexp[S,9:16])
ex.m   = cbind(rowMeans(ex.P), rowMeans(ex.S))
m.diff = mvfft(eq.m - ex.m)/sqrt(n)
eq.Pf  = mvfft(eq.P-eq.m[,1])/sqrt(n)
eq.Sf  = mvfft(eq.S-eq.m[,2])/sqrt(n)
ex.Pf  = mvfft(ex.P-ex.m[,1])/sqrt(n)
ex.Sf  = mvfft(ex.S-ex.m[,2])/sqrt(n)
fv11   = rowSums(eq.Pf*Conj(eq.Pf))+rowSums(ex.Pf*Conj(ex.Pf))/(2*(N-1))
fv12   = rowSums(eq.Pf*Conj(eq.Sf))+rowSums(ex.Pf*Conj(ex.Sf))/(2*(N-1))
fv22   = rowSums(eq.Sf*Conj(eq.Sf))+rowSums(ex.Sf*Conj(ex.Sf))/(2*(N-1))
fv21   = Conj(fv12)
# Equal Means
T2     = rep(NA, 512)
for (k in 1:512){
 fvk   = matrix(c(fv11[k], fv21[k], fv12[k], fv22[k]), 2, 2)
 dk    = as.matrix(m.diff[k,])
 T2[k] = Re((N/2)*Conj(t(dk))%*%solve(fvk,dk))  }
eF = T2*(2*p.dim*(N-1))/(2*N-p.dim-1)
par(mfrow=c(2,2))
freq = 40*(0:511)/n  # Hz
tsplot(freq, eF, col=5, xlab="Frequency (Hz)", ylab="F Statistic", main="Equal Means")
abline(h = qf(.999, 2*p.dim, 2*(2*N-p.dim-1)), col=8)
# Equal P
kd    = kernel("daniell",m);
u     = Re(rowSums(eq.Pf*Conj(eq.Pf))/(N-1))
feq.P = kernapply(u, kd, circular=TRUE)
u     = Re(rowSums(ex.Pf*Conj(ex.Pf))/(N-1))
fex.P =	kernapply(u, kd, circular=TRUE)
tsplot(freq, feq.P[1:512]/fex.P[1:512], col=5, xlab="Frequency (Hz)", ylab="F Statistic", main="Equal P-Spectra")
abline(h=qf(.999, 2*L*(N-1),  2*L*(N-1)), col=8)
# Equal S
u     = Re(rowSums(eq.Sf*Conj(eq.Sf))/(N-1))
feq.S = kernapply(u, kd, circular=TRUE)
u     = Re(rowSums(ex.Sf*Conj(ex.Sf))/(N-1))
fex.S =	kernapply(u, kd, circular=TRUE)
tsplot(freq, feq.S[1:512]/fex.S[1:512], col=5, xlab="Frequency (Hz)", ylab="F Statistic", main="Equal S-Spectra")
abline(h=qf(.999, 2*L*(N-1),  2*L*(N-1)), col=8)
# Equal Spectra
u      = rowSums(eq.Pf*Conj(eq.Sf))/(N-1)
feq.PS = kernapply(u, kd, circular=TRUE)
u      = rowSums(ex.Pf*Conj(ex.Sf)/(N-1))
fex.PS = kernapply(u, kd, circular=TRUE)
fv11   = kernapply(fv11, kd, circular=TRUE)
fv22   = kernapply(fv22, kd, circular=TRUE)
fv12   = kernapply(fv12, kd, circular=TRUE)
Mi     = L*(N-1); M = 2*Mi
TS     = rep(NA,512)
for (k  in 1:512){
det.feq.k = Re(feq.P[k]*feq.S[k] - feq.PS[k]*Conj(feq.PS[k]))
det.fex.k = Re(fex.P[k]*fex.S[k] - fex.PS[k]*Conj(fex.PS[k]))
det.fv.k  = Re(fv11[k]*fv22[k] - fv12[k]*Conj(fv12[k]))
log.n1    = log(M)*(M*p.dim);  log.d1 = log(Mi)*(2*Mi*p.dim)
log.n2    = log(Mi)*2 +log(det.feq.k)*Mi + log(det.fex.k)*Mi
log.d2    = (log(M)+log(det.fv.k))*M
r         = 1 - ((p.dim+1)*(p.dim-1)/6*p.dim*(2-1))*(2/Mi - 1/M)
TS[k]     = -2*r*(log.n1+log.n2-log.d1-log.d2)   }
tsplot(freq, TS, col=5, xlab="Frequency (Hz)", ylab="Chi-Sq Statistic", main="Equal Spectral Matrices")
abline(h = qchisq(.9999, p.dim^2))  # too small to be on plot

```



<br/> Example 7.10

```r
P = 1:1024; S = P+1024
mag.P  = log10(apply(eqexp[P,], 2, max) - apply(eqexp[P,], 2, min))
mag.S  = log10(apply(eqexp[S,], 2, max) - apply(eqexp[S,], 2, min))
eq.P   = mag.P[1:8];  eq.S = mag.S[1:8]
ex.P   = mag.P[9:16]; ex.S = mag.S[9:16]
NZ.P   = mag.P[17];   NZ.S = mag.S[17]
# Compute linear discriminant function
cov.eq = var(cbind(eq.P, eq.S))
cov.ex = var(cbind(ex.P, ex.S))
cov.pooled = (cov.ex + cov.eq)/2
means.eq   =  colMeans(cbind(eq.P, eq.S))
means.ex   =  colMeans(cbind(ex.P, ex.S))
slopes.eq  = solve(cov.pooled, means.eq)
inter.eq   = -sum(slopes.eq*means.eq)/2
slopes.ex  = solve(cov.pooled, means.ex)
inter.ex   = -sum(slopes.ex*means.ex)/2
d.slopes   = slopes.eq - slopes.ex
d.inter    = inter.eq - inter.ex
# Classify new observation
new.data   = cbind(NZ.P, NZ.S)
d          = sum(d.slopes*new.data) + d.inter
post.eq    = exp(d)/(1+exp(d))
# Print (disc function, posteriors) and plot results
cat(d.slopes[1], "mag.P +" , d.slopes[2], "mag.S +" , d.inter,"\n")
cat("P(EQ|data) =", post.eq,  "  P(EX|data) =", 1-post.eq, "\n" )
tsplot(eq.P, eq.S, xlim = c(0,1.5), ylim = c(.75,1.25), type='p', xlab = "log mag(P)", ylab = "log mag(S)",  pch = 8, cex=1.1, lwd=2, col=4, main="Classification Based on Magnitude Features")
 points(ex.P, ex.S, pch = 6, cex=1.1, lwd=2, col=6)
 points(new.data, pch = 3, cex=1.1, lwd=2, col=3) #rgb(0,.6,.2))
 abline(a = -d.inter/d.slopes[2], b = -d.slopes[1]/d.slopes[2])
 text(eq.P-.07,eq.S+.005, label=names(eqexp[1:8]), cex=.8)
 text(ex.P+.07,ex.S+.003, label=names(eqexp[9:16]), cex=.8)
 text(NZ.P+.05,NZ.S+.003, label=names(eqexp[17]), cex=.8)
 legend("topright", legend=c("EQ", "EX", "NZ"), pch=c(8,6,3), pt.lwd=2, cex=1.1, bg='white', col=c(4,6,3))
# Cross-validation
all.data = rbind(cbind(eq.P, eq.S), cbind(ex.P, ex.S))
post.eq <- rep(NA, 8) -> post.ex
for(j in 1:16) {
 if (j <= 8){samp.eq = all.data[-c(j, 9:16),]
  samp.ex = all.data[9:16,]}
 if (j > 8){samp.eq = all.data[1:8,]
  samp.ex = all.data[-c(j, 1:8),]   }
 df.eq      = nrow(samp.eq)-1;  df.ex = nrow(samp.ex)-1
 mean.eq    = colMeans(samp.eq);  mean.ex = colMeans(samp.ex)
 cov.eq = var(samp.eq);  cov.ex = var(samp.ex)
 cov.pooled = (df.eq*cov.eq + df.ex*cov.ex)/(df.eq + df.ex)
 slopes.eq  = solve(cov.pooled, mean.eq)
 inter.eq   = -sum(slopes.eq*mean.eq)/2
 slopes.ex  = solve(cov.pooled, mean.ex)
 inter.ex   = -sum(slopes.ex*mean.ex)/2
 d.slopes   = slopes.eq - slopes.ex
 d.inter    = inter.eq - inter.ex
 d          = sum(d.slopes*all.data[j,]) + d.inter
 if (j <= 8) post.eq[j] = exp(d)/(1+exp(d))
 if (j > 8) post.ex[j-8] = 1/(1+exp(d))  }
Posterior = cbind(1:8, post.eq, 1:8, post.ex)
colnames(Posterior) = c("EQ","P(EQ|data)","EX","P(EX|data)")
round(Posterior,3)  # Results from Cross-validation 

```




<br/> Example 7.11

```r
P = 1:1024; S = P+1024; p.dim = 2; n =1024
eq   = as.ts(eqexp[, 1:8])
ex   = as.ts(eqexp[, 9:16])
nz   = as.ts(eqexp[, 17])
f.eq <- array(dim=c(8, 2, 2, 512)) -> f.ex
f.NZ = array(dim=c(2, 2, 512)) 
# below calculates determinant for 2x2 Hermitian matrix
det.c <- function(mat){return(Re(mat[1,1]*mat[2,2]-mat[1,2]*mat[2,1]))}
L = c(15,13,5)      # for smoothing
for (i in 1:8){     # compute spectral matrices
 f.eq[i,,,] = mvspec(cbind(eq[P,i], eq[S,i]), spans=L, taper=.5, plot=FALSE)$fxx
 f.ex[i,,,] = mvspec(cbind(ex[P,i], ex[S,i]), spans=L, taper=.5, plot=FALSE)$fxx
 }
 u = mvspec(cbind(nz[P], nz[S]), spans=L, taper=.5, plot=FALSE)
 f.NZ = u$fxx	
bndwidth = u$bandwidth*40  # about .75 Hz
fhat.eq = apply(f.eq, 2:4, mean)    # average spectra
fhat.ex = apply(f.ex, 2:4, mean)
# plot the average spectra
par(mfrow=c(2,2))
Fr = 40*(1:512)/n
tsplot(Fr,Re(fhat.eq[1,1,]),col=5,xlab="Frequency (Hz)",ylab="",main="Average P-spectra")
tsplot(Fr,Re(fhat.eq[2,2,]),col=5,xlab="Frequency (Hz)",ylab="",main="Average S-spectra")
tsplot(Fr,Re(fhat.ex[1,1,]),col=5,xlab="Frequency (Hz)",ylab="")
tsplot(Fr,Re(fhat.ex[2,2,]),col=5,xlab="Frequency (Hz)",ylab="")
mtext("Earthquakes", side=2, line=-1, adj=.8, font=2, outer=TRUE)
mtext("Explosions", side=2, line=-1, adj=.2, font=2, outer=TRUE)
par(fig = c(.75, 1, .75, .98), new = TRUE)
ker = kernel("modified.daniell", L)$coef; ker = c(rev(ker),ker[-1])
plot((-33:33)/40, ker, type="l", ylab="", xlab="", cex.axis=.7, yaxp=c(0,.04,2))
# Choose alpha
Balpha = rep(0,19)
 for (i in 1:19){  alf=i/20
 for (k in 1:256) {  	
 Balpha[i]= Balpha[i] + Re(log(det.c(alf*fhat.ex[,,k] + (1-alf)*fhat.eq[,,k])/det.c(fhat.eq[,,k])) -   alf*log(det.c(fhat.ex[,,k])/det.c(fhat.eq[,,k])))} }
alf = which.max(Balpha)/20    # alpha = .4
# Calculate Information Criteria
rep(0,17) -> KLDiff -> BDiff -> KLeq -> KLex -> Beq -> Bex
for (i in 1:17){
 if (i <= 8) f0 = f.eq[i,,,]
 if (i > 8 & i <= 16) f0 = f.ex[i-8,,,]
 if (i == 17) f0 = f.NZ
for (k in 1:256) {    # only use freqs out to .25
 tr = Re(sum(diag(solve(fhat.eq[,,k],f0[,,k]))))
 KLeq[i] = KLeq[i] + tr + log(det.c(fhat.eq[,,k])) - log(det.c(f0[,,k]))
 Beq[i] =  Beq[i] + Re(log(det.c(alf*f0[,,k]+(1-alf)*fhat.eq[,,k])/det.c(fhat.eq[,,k])) - alf*log(det.c(f0[,,k])/det.c(fhat.eq[,,k])))
 tr = Re(sum(diag(solve(fhat.ex[,,k],f0[,,k]))))
 KLex[i] = KLex[i] + tr +  log(det.c(fhat.ex[,,k])) - log(det.c(f0[,,k]))
 Bex[i] = Bex[i] + Re(log(det.c(alf*f0[,,k]+(1-alf)*fhat.ex[,,k])/det.c(fhat.ex[,,k])) - alf*log(det.c(f0[,,k])/det.c(fhat.ex[,,k]))) }
KLDiff[i] = (KLeq[i] - KLex[i])/n
BDiff[i] =  (Beq[i] - Bex[i])/(2*n) }
x.b = max(KLDiff)+.1; x.a = min(KLDiff)-.1
y.b = max(BDiff)+.01; y.a = min(BDiff)-.01
dev.new()
tsplot(KLDiff, BDiff, type="n", xlim=c(x.a,x.b), ylim=c(y.a,y.b), cex=1.1,lwd=2, xlab="Kullback-Leibler Difference",ylab="Chernoff Difference", main="Classification Based on Chernoff and K-L Distances")
abline(h=0, v=0, lty=5, col=8)
points(KLDiff[1:8], BDiff[1:8], pch=8, cex=1.1, lwd=2, col=4)
points(KLDiff[9:16], BDiff[9:16], pch=6, cex=1.1, lwd=2, col=6)	   
points(KLDiff[17], BDiff[17], pch=3, cex=1.1, lwd=2, col=3)
legend("topleft", legend=c("EQ","EX", "NZ"), pch=c(8,6,3), pt.lwd=2, col=c(4,6,3))
abline(h=0, v=0, lty=2, col=8)
text(KLDiff[-c(1,2,3,7,14)]-.075, BDiff[-c(1,2,3,7,14)], label=names(eqexp[-c(1,2,3,7,14)]), cex=.7)
text(KLDiff[c(1,2,3,7,14)]+.075, BDiff[c(1,2,3,7,14)], label=names(eqexp[c(1,2,3,7,14)]), cex=.7)

```




<br/> Example 7.12

```r
library(cluster)
n=1024; P=1:n; S=P+n; p.dim=2 
eq = as.ts(eqexp[, 1:8])
ex = as.ts(eqexp[, 9:16])
nz = as.ts(eqexp[, 17])
f = array(dim=c(17, 2, 2, 512))
L = c(15, 15)       # for smoothing
for (i in 1:8){     # compute spectral matrices
 f[i,,,] = mvspec(cbind(eq[P,i], eq[S,i]), spans=L, taper=.5, plot=FALSE)$fxx
 f[i+8,,,] = mvspec(cbind(ex[P,i], ex[S,i]), spans=L, taper=.5, plot=FALSE)$fxx }
f[17,,,] = mvspec(cbind(nz[P], nz[S]), spans=L, taper=.5, plot=FALSE)$fxx	
JD = matrix(0, 17, 17)
# Calculate Symmetric Information Criteria
for (i in 1:16){
 for (j in (i+1):17){	
  for (k in 1:256) {    # only use freqs out to .25
    tr1 = Re(sum(diag(solve(f[i,,,k], f[j,,,k]))))
    tr2 = Re(sum(diag(solve(f[j,,,k], f[i,,,k]))))
    JD[i,j] = JD[i,j] + (tr1 + tr2 - 2*p.dim)}}}
 JD = (JD + t(JD))/n
colnames(JD) = c(colnames(eq), colnames(ex), "NZ")
rownames(JD) = colnames(JD)
cluster.2 = pam(JD, k = 2, diss = TRUE)
summary(cluster.2)  # print results (not shown)
par(mar=c(2,2,1,.5)+.5,  mgp = c(1.4,.6,0), cex=3/4, cex.lab=4/3, cex.main=4/3)
clusplot(JD, cluster.2$cluster, col.clus=gray(.5), labels=3, lines=0, main="Clustering Results for Explosions and Earthquakes", col.p=c(rep(4,8),rep(6,8), 3))   
text(-4.5,-.8, "Group I",  cex=1.1, font=2) 
text( 3.5,  5, "Group II", cex=1.1, font=2)

```




<br/> Example 7.13

```r
Per = mvspec(fmri1[,-1], plot=FALSE)$spec
par(mfrow=c(2,4)) 
for (i in 1:8){
 tsplot(ts(Per[1:21,i]), xaxt='n', nx=NA, ny=NULL,  minor=FALSE, col=5, ylim=c(0,8), main=colnames(fmri1)[i+1], xlab="Cycles", ylab="Periodogram" )
axis(1, at=seq(0,20,by=4))
abline(v=seq(0,20,by=4), col=gray(.9), lty=1)  }
dev.new() 
fxx = mvspec(fmri1[,-1], kernel=bart(2), taper=.5, plot=FALSE)$fxx
l.val = c()
for (k in 1:64) {
u = eigen(fxx[,,k], symmetric=TRUE, only.values = TRUE)
l.val[k] =  u$values[1]}  # largest e-value
tsplot(l.val, col=5, type='l', nx=NA, ny=NULL, minor=FALSE, xaxt='n',  xlab="Cycles (Frequency x 128)", ylab="First Principal Component")
axis(1, seq(4,60,by=8)); 
abline(v=seq(4,60,by=8), col=gray(.9) )
# At freq 4/128
u = eigen(fxx[,,4], symmetric=TRUE)
lam=u$values;  evec=u$vectors
lam[1]/sum(lam)          # % of variance explained
sig.e1 = matrix(0,8,8)
for (l in 2:5){          # last 3 evs are 0
 sig.e1 = sig.e1 + lam[l]*evec[,l]%*%Conj(t(evec[,l]))/(lam[1]-lam[l])^2}
 sig.e1 = Re(sig.e1)*lam[1]*sum(bart(2)$coef^2)
p.val = round(pchisq(2*abs(evec[,1])^2/diag(sig.e1), 2, lower.tail=FALSE), 3)
cbind(colnames(fmri1)[-1], abs(evec[,1]), p.val) # table values

```


<br/> Example 7.14

```r
bhat = sqrt(lam[1])*evec[,1]
(Dhat = Re(diag(fxx[,,4] - bhat%*%Conj(t(bhat)))))
(res = Mod(fxx[,,4] - Dhat - bhat%*%Conj(t(bhat))))

```



<br/> Example 7.15

```r
gr = diff(log(ts(econ5, start=1948, frequency=4))) # growth rate
tsplot(gr, ncol=2, col=2:6)
# scale each series to have variance 1
gr.s= scale(gr, center = FALSE, scale = apply(gr, 2, sd))
gr.spec = mvspec(gr.s, spans=c(7,7), taper=.25, lwd=2, col=2:6, lty=(1:6)[-3], main=NA) 
legend("topright", colnames(econ5), lty=(1:6)[-3], col=2:6, lwd=2, bg='white')
dev.new()
plot.spec.coherency(gr.spec, ci=NA, col=5, lwd=2, main=NA)
dev.new()
# PCs
n.freq = length(gr.spec$freq)
lam = matrix(0,n.freq,5)
for (k in 1:n.freq) lam[k,] = eigen(gr.spec$fxx[,,k], symmetric=TRUE, only.values=TRUE)$values 
par(mfrow=c(2,1))
tsplot(gr.spec$freq, lam[,1], col=5, ylab="", xlab="Frequency (\u00D7 4)", main="First Eigenvalue")
abline(v=.25, lty=5, col=8)
tsplot(gr.spec$freq, lam[,2], col=5, ylab="", xlab="Frequency (\u00D7 4)", main="Second Eigenvalue")
abline(v=.125, lty=5, col=8) 
e.vec1 = eigen(gr.spec$fxx[,,10], symmetric=TRUE)$vectors[,1]
e.vec2 = eigen(gr.spec$fxx[,,5], symmetric=TRUE)$vectors[,2]
round(Mod(e.vec1), 2);  round(Mod(e.vec2), 3)

```

<br/> Sleep pretty baby do not cry

```r
par(mfrow=2:1)
x = sleep1[[1]][,2]
tsplot(x, type='s', col=4, yaxt='n', ylab='', margins=c(0,.75,0,0)+.25)
 states = c('NR4', 'NR3', 'NR2', 'NR1', 'REM', 'AWAKE')
 axis(side=2, 1:6, labels=states, las=1)
 mtext('Sleep State', side=2, line=2.5, cex=1)
x = x[!is.na(x)]
mvspec(x, col=5, main=NA)
 abline(v=1/60, col=8, lty=5)
 mtext('1/60', side=1, adj=.04, cex=.75)

```


<br/> Example 7.17 

```r
xdata = dna2vector(bnrf1ebv)
u = specenv(xdata, spans=c(7,7), col=5)  # print u for details
dev.new()
id = c("(a)", "(b)", "(c)", "(d)")
par(mfrow=c(2,2))
for (j in 1:4){
  L = 1 + (j-1)*1000
  U = min(j*1000, length(bnrf1ebv))
  specenv(xdata, spans=c(7,7), section=L:U, col=5, ylim=c(0,1.28)) 
  text(.475, 1.25, id[j]) 
}

```

<br/> Example 7.18

```r
x     = astsa::nyse  # many packages have an 'nyse' data set
xdata = cbind(x, abs(x), x^2)
par(mfrow=2:1)
u = specenv(xdata, real=TRUE, col=5, spans=c(3,3))
# peak at freq = .001  
beta = u[2, 3:5]  # scalings
( b  = beta/beta[2] )  # makes abs(x) coef=1
gopt = function(x) { b[1]*x + b[2]*abs(x) + b[3]*x^2 }
x = seq(-.2, .2, by=.001)
tsplot(x, gopt(x), col=4, xlab='x', ylab='g(x)')
lines(x, abs(x), col=6)
legend('bottomright', lty=1, col=c(4,6), legend=c('optimal', 'absolute value'), bg='white')

```

[<sub>top</sub>](#table-of-contents)

---
