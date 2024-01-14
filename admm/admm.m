%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Copyright (C) 2020 Zaiwen Wen, Haoyang Liu, Jiang Hu
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 实例：交替方向乘子法（ADMM）解 LASSO 问题
% 考虑LASSO 问题 
% 
% $$\min_x \mu\|x\|_1+\frac{1}{2}\|Ax-b\|_2^2.$$
% 
% 首先考虑利用 ADMM 求解原问题：将其转化为 ADMM 标准问题
%
% $$ 
% \begin{array}{rl}
% \displaystyle\min_{x,z} & \hspace{-0.5em}\frac{1}{2}\|Ax-b\|^2_2+\mu\|z\|_1,\\
% \displaystyle\mathrm{s.t.} & \hspace{-0.5em}x=z,
% \end{array} 
% $$
%
% 则可以利用 ADMM 求解。相应的，对于 LASSO 对偶问题
%
% $$ 
% \begin{array}{rl}
% \displaystyle\min_y & \hspace{-0.5em}b^\top y+\frac{1}{2}\|y\|_2^2,\\
% \displaystyle\mathrm{s.t.} & \hspace{-0.5em}\|A^\top y\|_\infty\le\mu,
% \end{array}
% $$
%
% 则等价于
%
% $$ 
% \begin{array}{rl}
% \displaystyle\min_{y,z} & \hspace{-0.5em}b^\top y +\frac{1}{2}\|y\|_2^2+I_{\|z\|_\infty\le\mu}(z),\\
% \displaystyle\mathrm{s.t.} & \hspace{-0.5em}A^\top y + z = 0.
% \end{array}
% $$
%
% 对于上述的两个等价问题利用 ADMM 求解。
%% 构造 LASSO 问题
% 设定随机种子。
clear;
seed = 42;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 构造 LASSO 优化问题
% 
% $$\displaystyle\min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1.$$
% 
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。 正则化系数 $\mu=10^{-3}$。 随机迭代初始点。
m = 1024;
n = 2048;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = randn(n, 1);
mu = 1e-2;
%% 利用 ADMM 求解 LASSO 问题
% 首先在更严格的停机准则下进行试验，将收敛时得到的历史最优函数值作为真实的最优值的参考 $f^*$。
opts = struct();
opts.verbose = 0;
opts.maxit = 2000;
opts.sigma = 1e-2;
opts.ftol = 1e-12; 
opts.gtol = 1e-15;
[x, out] = LASSO_admm_primal(x0, A, b, mu, opts);
f_star = min(out.fvec); 
%%%

% 利用 ADMM 求解 LASSO 原问题。
opts = struct();
opts.verbose = 0;
opts.maxit = 2000;
opts.sigma = 1e-2;
opts.ftol = 1e-8; 
opts.gtol = 1e-10;
[x, out] = LASSO_admm_primal(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = length(data2);
%% 结果可视化
% 对每一步的目标函数值与最优函数值的相对误差进行可视化。
fig = figure;
semilogy(0:k2-1, data2, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('ADMM解原问题');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','admm.eps');


%% 解的稀疏性可视化
index = 1:length(x);
scatter(index, x);
xlim([0 length(x)]);
xlabel("数组index");
ylabel("数组值");
title("解的稀疏性可视化");
saveas(gcf, "sparse.png");

%% 矩阵方程约束违反度曲线



%% x-z约束违反度曲线