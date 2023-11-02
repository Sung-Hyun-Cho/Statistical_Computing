
\title{Proximal Algorithms for SQRT-Lasso
Optimization\vspace{-0.5em}}
\date{}
\author{조성현, 이대하, 이유은}
\begin{document}
\maketitle


고차원의 sparse linear model $y = X\theta + \epsilon$을 고려하자.
$X \in \mathbb{R}^{n \times d}$ 는 design matrix, $y \in \mathbb{R}^n$은 response, $\epsilon \sim N(0,\sigma^2 I_n)$는 random noise, $\theta$는 sparse regression coefficient vector이다. $\theta$를 추정하는 한 방법으로 LASSO(Least Absolute Shrinkage and Selection Operator) 가 있으며 Lasso 에 의한 $\theta$의 추정량은
$$\bar{\theta}^{Lasso} = \argmin_{\theta \in \mathbb{R}^d} \frac{1}{n} \lVert y-X\theta \rVert_2^2 + \lambda_{Lasso}\lVert \theta \rVert_1$$
로 표현된다. 우변의 $L_1$-norm 항은 convex하기 때문에 minimize하기 용이하면서 sparse한 최적해를 만들어내는 장점이 있다.
특히 $\lambda$가 클 수록 $\Bar{\theta}^{Lasso}$ 의 sparsity가 증가하여 regression에 있어 유의미한 변수를 골라내는, 이른바 변수 선택에 있어 이점이 있는 방법이다. 따라서 Lasso 에 있어 유의미한 분석을 위한 적절한 $\lambda$의 선택은 아주 중요하다.\;\cite{hastie2015statistical}로부터 $\lambda_{Lasso}\asymp \sigma \sqrt{\frac{\log d}{n}}$ 임이 알려져 있으며, $\sigma$ 가 알려져있지 않은 parameter이기 때문에 최적의 $\lambda$를 찾는 것은 쉬운 과정이 아니다. 이에 대한 해결책으로 아래의 SQRT-Lasso 모형을 제안할 수 있으며,
$$
\Bar{\theta}^{SQRT} = \argmin_{\theta \in \mathbb{R}^d} \mathcal{F}_\lambda(\theta), \quad
\mathcal{F}_\lambda(\theta) = \mathcal{L}(\theta) + \lambda_{SQRT} \lVert \theta \rVert _1 , \quad 
\mathcal{L}(\theta) = \frac{1}{\sqrt{n}} \lVert y - X\theta \rVert _2 . 
$$
$\lambda_{SQRT}\asymp \sqrt{\frac{\log d}{n}}$ 임이 알려져있고 이를 이용하면 $\sigma$에 대한 정보 부족의 문제에서 벗어날 수 있다. 즉, 최적의 $\lambda$를 빠르게 찾음으로서 더 좋은 성능의 regression를 기대할 수 있다.
\subsection{기존 방법론의 개선점}
$l_2$ loss 자체가 전역 미분가능하지도 않고, 그 gradient가 Lipschitz 연속이라고도 볼 수 없기 때문에 least square loss보다 다루기가 어렵다. 기존의 방법들도 data matrix의 차원이 작을 때에만 적합하다. 다만 이 함수의 구조적 특성을 이용하면 empirical하게 좋은 성능을 보이는 방법을 고려할 수 있다. 본 연구에서는 이러한 근사 알고리즘을 주제로 다룬다.
\vspace{-1.5em}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Algorithm\vspace{-0.5em}}
SQRT-Lasso에서의 $\mathcal{L}(\theta)$는 $||y-X\theta|| = 0$인 $\theta$에 대해서 미분 불가능한 함수지만, $\lambda$를 충분히 큰 값으로 설정하면 $\mathcal{F}$을 minimize하는 $\theta$가 이러한 nonsmooth한 영역 밖에 위치하게 되므로 $\mathcal{L}(\theta)$를 미분가능한 함수로서 취급할 수 있다. 이 때문에 SQRT-Lasso 모형에 대해 목적함수 $\mathcal{L}(\theta)$에 근사하는 quadratic polynomial로 대체해 최적화 문제를 해결하는 Proximal Algorithm 을 고려할 수 있다. 본 연구에서 사용될 Proximal Gradient Descent Algorithm (Prox-GD) 알고리즘은 다음과 같이 $\mathcal{F}_{\lambda}(\theta)$를 근사한다.
\begin{equation}
    \mathcal{Q}_{\lambda}(\theta, \theta^{(t)}) = \mathcal{L}(\theta^{(t)}) + \nabla\mathcal{L}(\theta^{(t)})^{T}(\theta-\theta^{(t)}) + \frac{L^{(t)}}{2}||\theta - \theta^{(t)}||_{2}^{2} + \lambda||\theta||_{1} \ \ (Prox-GD) \\
\end{equation}
Prox-GD 알고리즘의 경우 2차 항의 계수인 $\mathcal{L}^{(t)}$ 는 backtracking line search를 통해 결정지을 수 있다. 이후 $\argmin_{\theta}{\mathcal{Q}_{\lambda}(\theta, \theta^{(t)})} = S_{\frac{\lambda}{L^{(t)}}}(\theta^{(t)}-\frac{\nabla\mathcal{L}(\theta^{(t)}}{L^{(t)}})$ 임을 이용하여 반복적으로 $\theta$를 근사해 나갈 수 있다. $S_{\lambda}(x)=[sign(x_{j})max\{|x_{j}|-\lambda,0\}]_{j=1}^{d}$ 로 정의한다. \\
한편, Proximal Newton Algorithm(Prox-Newton) 알고리즘의 경우 다음과 같은 근사를 사용한다. 
\begin{equation}
    \mathcal{Q}_{\lambda}(\theta, \theta^{(t)}) = \mathcal{L}(\theta^{(t)}) + \nabla\mathcal{L}(\theta^{(t)})^{T}(\theta-\theta^{(t)}) + \frac{1}{2}||\theta - \theta^{(t)}||_{\nabla^{2}\mathcal{L}(\theta^{(t)})}^{2} + \lambda||\theta||_{1} \ \ (Prox-Newton)
\end{equation}
위 식에서 $||\theta - \theta^{(t)}||_{\nabla^{2}\mathcal{L}(\theta^{(t)})}^{2} = (\theta-\theta^{(t)})^{T}\nabla^{2}\mathcal{L}(\theta^{(t)})(\theta-\theta^{(t)})$를 의미한다. Prox-Newton 알고리즘의 경우 $\mathcal{Q}_{\lambda}(\theta, \theta^{(t)})$를 최소로 하는 $\theta^{(t+0.5)}$를 구한 후, 마찬가지로 backtracking line search를 통해 결정될 적절한 step size $\eta_{t}$에 대해 $\theta^{(t+1)}=\theta^{(t)}+\eta_{t}(\theta^{(t+0.5)}-\theta^{(t)})$ 으로 더 작은 $\mathcal{F}_{\lambda}(\theta)$ 값을 갖는 $\theta$를 알고리즘이 종료되기 전까지 반복적으로 찾아줄 수 있다. \\
두 알고리즘은 미리 정해 둔 stopping criterion $\epsilon$ 값보다 $\omega_{\lambda}(\theta^{(t)})=\min_{g\in\partial||\theta^{(t)}||_{1}}{||\nabla\mathcal{L}(\theta^{(t)})+\lambda g||_{\infty}}$가 더 작아질 때 종료되는 것으로 한다. \\
\indent regularization parameter를 보다 높게 설정할수록 해의 움직임은 smooth한 영역에서 이루어지게 된다. 따라서 초기에는 $\lambda$를 높게 설정한 후 stage별로 차츰 줄여가면서 최종적으로 원하는 값으로 맞추는 방법을 고려할 수 있다. 초기값으로는 all-zero 해에 대응되는 $\lambda_{[0]} = {||\nabla\mathcal{F}_\lambda(0)||}_{\infty}$을 설정하고, $\lambda_{[N]} = \sqrt{\frac{\log d}{n}}$, 그리고 그 사이 단계에서는 등비수열을 이루도록 $\lambda_{[i+1]} = (\frac{\lambda_{[N]}}{\lambda_{[0]}})^{\frac{1}{N}} \lambda_{[i]}$ 으로 둔다. 각 단계에서 $\theta$의 초기값으로는 각 단계의 종료 시점에서의 값을 사용한다. \\
이러한 pathwise optimization을 통해서, 이론적으로는 fast convergence region에 solution이 진입하게 됨이 근거된다. 또한, 진입 이전에는 regularization parameter가 진입 이후보다 더 크기 때문에 설령 진입하기 이전이라도 nonsmooth region에서 벗어나 있음을 보장할 수 있다. 따라서 준 proximal algorithm에 실제로 빠른 수렴성을 확보해줄 수 있게 된다.

\vspace{-1.5em}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Evaluation\vspace{-0.5em}}
\cite{Xingguo2020Fast}에서 제시한 알고리즘은 pathwise optimization scheme을 이용해 빠르게 수렴에 이르러, loss function의 nonsmooth한 영역에 들어가는 것을 방지한다. 따라서 알고리즘의 성능을 측정하기 이전에 이에 대한 확인이 우선적으로 이루어져야 한다. 본 연구에서는 기지의 분포를 따르는 train dataset을 생성하고, optimization의 각 단계마다 $\mathcal{F}_\lambda(\theta)$ 값을 관찰하여 global optimum 주변에서 얼마나 빠르게 수렴하는지 iteration 횟수와 비교하여 관찰할 것이다. \cite{Xingguo2020Fast}에서 제시한 linear convergence(Prox-GD) 와 quadratic convergence(Prox-Newton) 의 결과와 비교하는 것에 의미를 둔다. \\
또한, 알고리즘의 성능을 평가하기 위해서 synthetic dataset을 다양한 조건 하에서 생성한 후 training time을 측정하여 비교한다. 주된 변인으로는 error variance와 stopping criterion, 그리고 pathwise optimization에서의 step 수를 사용한다. 한편 SQRT-Lasso를 통해 overfitting이 충분히 방지되었는지 확인하기 위해서, 각 iteration stage에서의 $\widehat \theta$ 에 대한 mean square error 값들의 minimum을 함께 계산한다. \\
마지막으로 실제 dataset을 사용하여, 유사한 suboptimality를 보장하는 조건에서 SQRT-Lasso를 근사하는 다른 알고리즘과 준 알고리즘의 수행 시간을 비교한다. 비교군으로는 ADMM, ScalReg, CD 등을 사용할 수 있다. dataset에 대해서는 다음 section에서 더 자세히 설명한다. \\

\vspace{-1.5em}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Dataset\vspace{-0.5em}}
Proximal algorithm을 이용하여 SQRT-Lasso 모형에 적합한 분류를 진행하기 위해 design matrix의 sample의 수보다 feature의 수가 더 많은 경우로 제한하였다. 이에 801개의 sample, 20531개의 feature로 구성된 'gene expression cancer RNA-Seq' 데이터를 사용하고자 한다 \cite{misc_gene_expression_cancer_rna-seq_401}. 이 데이터는 sample을 BRCA(Breast cancer susceptibility gene), KIRC(Kidney renal clear cell carcinoma), COAD(Colon adenocarcinoma), LUAD(Lung adenocarcinoma), PRAD(Prostate adenocarcinoma)의 5개 종류의 tumor로 분류하였다. 본 연구에서는 우선 response가 vector인 경우로 제한하여, BRCA인 경우만 1, 나머지 경우는 0으로 SQRT-Lasso regression을 진행하여 알고리즘의 수행 시간을 확인해보고자 한다. 나아가 각 sample의 label에 따라 BRCA이면 $[1, 0, 0, 0]$, KIRC이면 $[0, 1, 0, 0]$, COAD이면 $[0, 0, 1, 0]$, LUAD이면 $[0, 0, 0, 1]$, PRAD이면 $[0, 0, 0, 0]$의 row vector를 대응시켜 CMR(calibrated multivariate regression)으로 준 알고리즘을 활용해보는 것을 목표로 한다. \\

\vspace{-1.5em}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{역할 분담}
\noindent
조성현: fast convergence 확인, 알고리즘 수행 시간 비교, CMR으로의 활용 \\
이대하:  synthetic data 생성, real data에서 알고리즘 적용, 비교군 알고리즘 구현 \\
이유은: Prox-GD/Prox-Newton 구현, pathwise optimization 구현 \\

\bibliographystyle{plain}
\bibliography{references}
\end{document}
**
