# 机器学习中的优化算法期末大作业

> 2019011831 项晨东



### 问题分析

作业需要解决的是$ℓ0$范数最小化问题。
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
在这了我们使用扩展的MCP函数，其函数定义为：

$$
MCP_{\lambda, \gamma}\left(u_i\right)= \begin{cases}\lambda\left|u_i\right|-\frac{u_i^2}{2 \gamma}, & \text { if }\left|u_i\right| \leq \gamma \lambda, \\ \frac{1}{2} \gamma \lambda^2, & \text { if }\left|u_i\right|>\gamma \lambda,\end{cases}
$$

(3) $\mathrm{SCAD}$ 函数 $\left(\right.$ 参数 $\left.\gamma_2>\gamma_1>0\right)$
$$
g(\theta)= \begin{cases}|\theta|, & |\theta| \leq \gamma_1, \\ \frac{2 \gamma_2|\theta|-\theta^2-\gamma_1^2}{2\left(\gamma_2-\gamma_1\right)}, & \gamma_1<|\theta| \leq \gamma_2, \\ \frac{\gamma_1+\gamma_2}{2}, & |\theta| \geq \gamma .\end{cases}
$$


注意到这些松弛函数有以下的特点

- 松弛函数带一个常数参数$\gamma$, 函数效果收到$\gamma$的影响。
- 从语义上理解, 松弛函数惩罚哪些绝对值小的值向0优化, 而不惩罚那些绝对值大于$\gamma$的值。

#### 函数选择
在本次作业中, 我们使用MCP函数作为松弛函数进行优化。


在本次作业中，$\lambda$始终为1。
$$
\text{MCP}'(x; \gamma,\lambda) = 
 \begin{cases} 
 1 - \frac{x}{\gamma}, & \text{if } x > 0 \text{ and } x \leq \gamma, \\ 
-1 - \frac{x}{\gamma}, & \text{if } x < 0 \text{ and } -\gamma \leq x, \\ 
 0, & \text{if } |x| > \gamma. 
 \end{cases}
$$

使用组合优化的方法来看待问题，则问题转化为：
$$
\begin{aligned}
\min & \frac{1}{2}\|A x -b\|_2^2 + \mu MCP_{\gamma,\lambda}(z) \\
\text { s.t. } & x=z .
\end{aligned}
$$


写出增广的Lagrangian函数
$L_\rho(x,z,y)=\frac{1}{2}\|Ax-b\|_2^2+\mu MCP_{\gamma,\lambda}(z)+y^\top(x-z)+\frac{\rho}{2}\|x-z\|_2^2$。


相对于LASSO问题的admm算法，唯一需要修改的就是对于z的迭代方式
$z^{k+1}=\arg\min_z 
\left(\mu MCP_{\gamma,\lambda}(z)+\frac{\sigma}{2}\|x^{k+1}-z+y^k/\sigma\|_2^2\right)$
或者
$z^{k+1}=\arg\min_z 
\left(MCP_{\gamma,\lambda}(z)+\frac{\sigma}{2 \mu}\|x^{k+1}-z+y^k/\sigma\|_2^2\right)$

#### 求解简化版本最优值
分析简化版此函数的最优解, 此处z为单值而非向量：
$\arg\min_z 
\left(MCP_{\gamma,\lambda}(z)+\frac{\rho}{2}\|s-z\|_2^2\right)$

如果$\rho > \frac{1}{\gamma}$:
$$
z= \begin{cases}s, & \text { if }\left|s\right|>\gamma \lambda \\
 \operatorname{sign}\left(s\right) \frac{\left|s\right|-\frac{\lambda}{\rho}}{1-\frac{1}{\gamma \rho}}, & \text { if } \frac{\lambda}{\rho}<\left|s_i\right| \leq \gamma \lambda \\ 
 0, & \text { if }\left|s\right| \leq \frac{\lambda}{\rho}\end{cases}
$$
如果$\rho = \frac{1}{\gamma}$:
$$
z= \begin{cases}s, & \text { if }\left|s\right|>\gamma \lambda \\ 0, & \text { if }\left|s\right| \leq \gamma \lambda\end{cases}
$$
如果$\rho < \frac{1}{\gamma}$:
$$
z= \begin{cases}s, & \text { if }\left|s\right|>\sqrt{\frac{\gamma}{\rho}} \lambda \\ 0, & \text { if }\left|s\right| \leq \sqrt{\frac{\gamma}{\rho}} \lambda\end{cases}
$$

在实际的解题过程中，以上都可用以下的式子做近似：
$$
z= \begin{cases}s_i, & \text { if }\left|s\right|>\gamma \lambda \\ \operatorname{sign}\left(s\right) \frac{\left|s\right|-\lambda}{1-\frac{1}{\gamma}}, & \text { if } \lambda<\left|s\right| \leq \gamma \lambda \\ 0, & \text { if }\left|s\right| \leq \lambda\end{cases}
$$

则$z^{k+1}=\arg\min_z 
\left(MCP_{\gamma,\lambda}(z)+\frac{\sigma}{2 \mu}\|x^{k+1}-z+y^k/\sigma\|_2^2\right)$迭代公式为：

$$
z_i^{k+1}= 
\begin{cases}
\frac{S\left(x_i^{k+1}+\frac{y_i^k}{\sigma}, \mu \lambda\right)}{1- \mu / \gamma}, & i f\left|x_i^{k+1}+\frac{y_i^k}{\sigma}\right| \leq \gamma \lambda \\ 
x_i^{k+1}+\frac{y_i^k}{\sigma}, & \text { if }\left|x_i^{k+1}+\frac{y_i^k}{\sigma}\right|>\gamma \lambda
\end{cases}
$$
其中S为
$$
S(z, \eta)=\operatorname{sign}(z) \max \{|z|-\eta, 0\}
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








#### 交替方向乘子法

唯一需要修改的事对于z的更新

$z^{k+1}=\arg\min_z 
\left(\mu MCP(z)+\frac{\sigma}{2}\|x^{k+1}-z+y^k/\sigma\|_2^2\right)$

根据一阶条件可知：

$$
z^{k+1} =
\begin{cases} 
\frac{-\mu + y^{k} + \sigma x^{k+1} }{\sigma - \frac{\mu}{\gamma}}, & \text{if } z^{k+1} > 0 \text{ and } z^{k+1} \leq \gamma, \\ 
\frac{\mu + y^{k} + \sigma x^{k+1} }{\sigma - \frac{\mu}{\gamma}}, & \text{if } z^{k+1} < 0 \text{ and } -\gamma \leq z^{k+1}, \\
x^{k+1} + y^{k}/\sigma, & \text{if } |z^{k+1}| > \gamma. 
 \end{cases}
 $$

 则可以根据条件求解。