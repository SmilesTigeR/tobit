# Tobit Regression
  
## Formation

\begin{aligned} Y^{*} = X^{'}\beta+\varepsilon \end{aligned}  
  
\begin{aligned} Y = \begin{cases} a, & \mbox{if } Y^{*} \le a \\ Y^{*}, & \mbox{if } a < Y^{*} < b \\ b, & \mbox{if } Y^{*} \ge b \end{cases} \end{aligned}  
  
where,  
- $X=\begin{pmatrix} x_{1} \\ \vdots \\ x_{p} \end{pmatrix}$
- $\beta=\begin{pmatrix} \beta_{1} \\ \vdots \\ \beta_{p} \end{pmatrix}$
- $\varepsilon \sim N(0, \sigma^2)$
  
## Likelihood Function
  
\begin{aligned} L = \prod_{i=1}^n{[I_{i}^{a}\Phi(\frac{a-X_{i}^{'}\beta}{\sigma})] \times [I_{i}^{b}(1-\Phi(\frac{b-X_{i}^{'}\beta}{\sigma}))] \times [(1-I_{i}^{a}-I_{i}^{b})\frac{1}{\sigma}\phi(\frac{y_{i}-X_{i}^{'}\beta}{\sigma})}] \end{aligned}  
  
where,  
- $I_{i}^{a}=\begin{cases} 1, & \mbox{if } y_{i} = a \\ 0, & \mbox{if } y_{i} > a \end{cases}$
- $I_{i}^{b}=\begin{cases} 1, & \mbox{if } y_{i} = b \\ 0, & \mbox{if } y_{i} < b \end{cases}$
- $X_{i}=\begin{pmatrix} x_{i1} \\ \vdots \\ x_{ip} \end{pmatrix}$
- $\phi(\cdot)$ is standard normal probability density function (PDF)
- $\Phi(\cdot)$ is standard normal cumulative density function (CDF)
  
## Log-likelihood Function
  
\begin{aligned} l = \log{L} = \sum_{i=1}^{n}{I_{i}^{a}\log{\Phi(\frac{a-X_{i}^{'}\beta}{\sigma})+I_{i}^{b}\log{(1-\Phi(\frac{b-X_{i}^{'}\beta}{\sigma}))}+(1-I_{i}^{a}-I_{i}^{b})(\log{\phi(\frac{y_{i}-X_{i}^{'}\beta}{\sigma})}-\frac{1}{2}\log{\sigma^2})}} \end{aligned}  
  
## Score Function (First Derivation)
  
\begin{aligned} \frac{\partial{l}}{\partial{\beta}}=\sum_{i=1}^{n}{(-I_{i}^{a}\frac{\phi(\frac{a-X_{i}^{'}\beta}{\sigma})}{\Phi(\frac{a-X_{i}^{'}\beta}{\sigma})}+I_{i}^{b}\frac{\phi(\frac{b-X_{i}^{'}\beta}{\sigma})}{1-\Phi(\frac{b-X_{i}^{'}\beta}{\sigma})}+(1-I_{i}^{a}-I_{i}^{b})\frac{y_{i}-X_{i}^{'}\beta}{\sigma})\frac{X_{i}^{'}}{\sigma}} \end{aligned} 
\begin{aligned} \frac{\partial{l}}{\partial{\sigma^2}}=\frac{1}{2\sigma}\frac{\partial{l}}{\partial{\sigma}}=\sum_{i=1}^{n}{\frac{1}{2\sigma^2}(-I_{i}^{a}\frac{\phi(\frac{a-X_{i}^{'}\beta}{\sigma})}{\Phi(\frac{a-X_{i}^{'}\beta}{\sigma})}\frac{a-X_{i}^{'}\beta}{\sigma}+I_{i}^{b}\frac{\phi(\frac{b-X_{i}^{'}\beta}{\sigma})}{1-\Phi(\frac{b-X_{i}^{'}\beta}{\sigma})}\frac{b-X_{i}^{'}\beta}{\sigma}+(1-I_{i}^{a}-I_{i}^{b})((\frac{y_{i}-X_{i}^{'}\beta}{\sigma})^2-1))} \end{aligned}  
  
## Hessian Matrix (Second Derivation)
  
\begin{aligned} -\frac{\partial{l}}{\partial{\beta}\partial{\beta^{'}}}=\sum_{i=1}^{n}{\frac{X_{i}}{\sigma}(I_{i}^{a}\frac{\phi(\frac{a-X_{i}^{'}\beta}{\sigma})(\Phi(\frac{a-X_{i}^{'}\beta}{\sigma})\frac{a-X_{i}^{'}\beta}{\sigma}+\phi(\frac{a-X_{i}^{'}\beta}{\sigma}))}{(\Phi(\frac{a-X_{i}^{'}\beta}{\sigma}))^{2}}-I_{i}^{b}\frac{\phi(\frac{b-X_{i}^{'}\beta}{\sigma})((1-\Phi(\frac{b-X_{i}^{'}\beta}{\sigma}))\frac{b-X_{i}^{'}\beta}{\sigma}-\phi(\frac{b-X_{i}^{'}\beta}{\sigma}))}{(1-\Phi(\frac{b-X_{i}^{'}\beta}{\sigma}))^{2}}+(1-I_{i}^{a}-I_{i}^{b}))\frac{X_{i}^{'}}{\sigma}} \end{aligned} 
  
## Prediction
\begin{aligned} E(Y|X)=\Phi(\frac{a-X_{i}^{'}\beta}{\sigma})\times a+(1-\Phi(\frac{b-X_{i}^{'}\beta}{\sigma}))\times b+(\Phi(\frac{b-X_{i}^{'}\beta}{\sigma})-\Phi(\frac{a-X_{i}^{'}\beta}{\sigma}))\times (X_{i}^{'}\beta-\sigma\frac{\phi(\frac{b_{i}-X_{i}^{'}\beta}{\sigma})-\phi(\frac{a_{i}-X_{i}^{'}\beta}{\sigma})}{\Phi(\frac{b-X_{i}^{'}\beta}{\sigma})-\Phi(\frac{a-X_{i}^{'}\beta}{\sigma})})\end{aligned}