# 机器学习中的优化算法期末大作业

> 2019011831 项晨东



### 问题分析

作业需要解决的是$$ℓ0$$范数最小化问题。
$$
\begin{aligned}
\min & \|x\|_0 \\
\text { s.t. } & A x=b .
\end{aligned}
$$
如果使用$ℓ0$范数松弛, 就是LASSO问题, 本题使用的是其他的松弛方法, 使用的松弛函数有如下三个:





(1) Capped $\ell_1$ 函数 $($ 参数 $\gamma>0)$
$$
g(\theta)= \begin{cases}|\theta|, & |\theta| \leq \gamma \\ \gamma, & |\theta| \geq \gamma\end{cases}
$$


(2) MCP函数 $($ 参数 $\gamma>0)$
$$
g(\theta)= \begin{cases}|\theta|-\frac{\theta^2}{2 \gamma}, & |\theta| \leq \gamma \\ \frac{\gamma}{2}, & |\theta| \geq \gamma .\end{cases}
$$


(3) $\mathrm{SCAD}$ 函数 $\left(\right.$ 参数 $\left.\gamma_2>\gamma_1>0\right)$
$$
g(\theta)= \begin{cases}|\theta|, & |\theta| \leq \gamma_1, \\ \frac{2 \gamma_2|\theta|-\theta^2-\gamma_1^2}{2\left(\gamma_2-\gamma_1\right)}, & \gamma_1<|\theta| \leq \gamma_2, \\ \frac{\gamma_1+\gamma_2}{2}, & |\theta| \geq \gamma .\end{cases}
$$


注意到这些松弛函数有以下的特点

- 松弛函数带一个常数参数$\gamma$, 函数效果收到$\gamma$的影响。
- 从语义上理解, 松弛函数惩罚哪些绝对值小的值向0优化, 而不惩罚那些绝对值大于$\gamma$的值。

在本次作业中, 我们使用MCP函数作为松弛函数进行优化。




$$
\text{MCP}'(x; \theta) = 
 \begin{cases} 
 1 - \frac{x}{\theta}, & \text{if } x > 0 \text{ and } x \leq \theta, \\ 
-1 - \frac{x}{\theta}, & \text{if } x < 0 \text{ and } -\theta \leq x, \\ 
 0, & \text{if } |x| > \theta. 
 \end{cases}
$$


### 算法实现和数值实验

最优性条件:

可知MCP函数仅在0点是不可导的, 在其他地方是连续的, 而残差项是连续的。

 该问题可以看成连续的复合优化问题, 可知符合优化问题的一阶必要条件可以从教材的定理中得出:

定理 5.6 (复合优化问题一阶必要条件) 令 $x^*$ 为问题 (5.3.1) 的一个局部极小点, 那么
$$
-\nabla f\left(x^*\right) \in \partial h\left(x^*\right),
$$

其中 $\partial h\left(x^*\right)$ 为凸函数 $h$ 在点 $x^*$ 处的次梯度集合.





虽然目标函数是非凸的, 我们依然可以尝试使用基于梯度的方法进行优化。







