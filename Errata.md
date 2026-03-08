## Edition 5 Errata  &#127379;



<br/>

### Chapter 1 

- it's perfect

### Chapter 2 

&#129300; A note on __Example 2.4__ and the __Lotka-Volterra (LV) equations__ because some people have asked. The example is included for two reasons, the main one being that we wanted an example of lagged regression that would also carry over into Section 3.8 (Regression with Autocorrelated Errors). Second was the ubiquitous  use of self-exciting threshold (SETAR) models for Lynx populations, which is ridiculous because without prey, a predator population would die out... there is nothing self-exciting in the process (unless they are all [cannibal lynx](https://www.reddit.com/r/HardcoreNature/comments/10l2x6a/a_canada_lynx_and_her_cub_scavenge_the_carcass_of/#:~:text=Indeed%2C%20I've%20never%20really%20seen%20any%20videos,It%20goes%20to%20show%20just%20how%20harsh) as shown in this video). We wanted to show that mathematical biologists have dealt with this problem in more reasonable ways.  

&#49;&#65039;&#8419;   The actual discrete-time LV equations  are 

$$ \nabla H_{t+1} = \alpha H_t + \beta L_t H_t $$

$$ \nabla L_{t+1} = \delta L_t + \gamma L_t H_t$$

_where_ $\alpha > 1$  _is the growth rate of the prey in the absence of the predator,_ $0 < \delta < 1$ _is the survival rate of the predator in the absence of its prey source_ ...

&#50;&#65039;&#8419; Unfortunately, $\nabla X_{t}$ wasn't defined yet (it's in the next section), so we just added $H_t$ / $L_t$ to each side, so the LV equations are 

$$ H_{t+1} = (\alpha + 1) H_t + \beta L_t H_t $$

$$  L_{t+1} = (\delta+1) L_t + \gamma L_t H_t$$
 
with the same definitions for the constants.  So the way it's in the text [equation (2.22)], the constants text-$\alpha$ and text-$\delta$ should be adjusted appropriately - they were not and that's the blooper.  

&#51;&#65039;&#8419; But wait, there's more.  The example fits the predator equation (for the lynx $L_{t}$) as given by the LV equations. Statisticians learn that you _should have all main effects if there are interactions_, so some people might have a hard time with this example.   However, this is a case where you have theoretical justification because if you include all main effects, you lose the cyclic nature of the LV equations. We recommend this video for students who haven't been exposed to diffeqs: <a href="https://www.youtube.com/watch?v=DDEvlLa9z_U" target="new"> Lotka-Volterra Equations</a> &#128076;



###  Chapter 3 

- perfect


### Chapter 4 

&#128064; a little blemish &hellip; on page 181, section 4.1, before (4.7) should be:

&hellip; then an estimate of &hellip; $\sigma_k^2$  would be the sample variance $S_k^2 = \frac{1}{2} (a_k^2+b_k^2)$ &hellip;

and (4.7) should be: 

$$\hat\gamma_x(0) = \widehat {\rm var}( x_t)= \tfrac{1}{2} \sum_{k=1}^q (a_k^2 +b_k^2).  \qquad\qquad    (4.7)$$

<br/>

 &#128064; Also, thanks to Professor Chris Koen, Dept Statistics, University of the Western Cape for noticing this one - in Problem 4.17: _&hellip; a peak at a quefrency corresponding to 1/D_ should be _&hellip; a peak at a quefrency corresponding to D_.

 > It's a mixed up, muddled up, shook up world, except for Lola


### Chapter 5 


- perfect


### Chapter 6 


- perfect


### Chapter 7 

- perfect

### Elsewhere

 - FYI: In Edition 5, Appendix R has been removed and put online  here: [dsstoffer.github.io/Rtoot](https://dsstoffer.github.io/Rtoot)


<br/><br/><br/>
