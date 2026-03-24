---
tags:
  - 原子笔记
type: concept
status: draft
created: 2026-03-18
updated: 2026-03-24
aliases:
  - DDPM,
---
# [[Denoising Diffusion Probabilistic Models]]

> 我们想学习到一个模型, 能够**从随机噪声出发**, 最后**生成**和**真实数据分布** $p_\mathrm{data}$ 一样的样本
> 
> DDPM采用以下想法:
> 1.  真实数据为 $x_0\sim p_\mathrm{data}$
> 2. **设计**一个**前向加噪**过程, 把 $x_0$ 一步一步加噪, 最后变成接近于高斯噪音的 $x_T$
> 3. **学习**一个**反向去噪**过程, 从 $x_T$ 一步一步去噪回到 $x_0$

# 核心理解 

- **解决的问题 ：** 
   
- **作用 /重要性：**
- **易错点/注意事项：**

# 前向加噪

## 前向加噪转移核

- DDPM 采用 Gaussian 形式 表示 从 第$k-1$步 到 第$k$步 的前向加噪转移核 $$q(x_k\mid x_{k-1})=\mathcal{N}(x_k;\sqrt{\alpha_k}\:x_{k-1},\:\beta_kI)\:,\quad\alpha_k=1-\beta_k$$
- 更常见的写法为:$$x_k=\sqrt{\alpha_k}x_{k-1}+\sqrt{\beta_k}\:\epsilon,\quad\epsilon\sim\mathcal{N}(0,I)$$
- 即:
	- 旧信号 $x_{k-1}$​ 被缩小一点, 其中 $\sqrt{\alpha_k}$ 表示信号的缩小程度
	- 再加入一点 Gaussian 噪声, 其中 $\beta_k$ 是每一步的加噪强度, 通常很小

## 前向多步分布

> 在DDPM中, 前向加噪过程被定义为一个马尔科夫链, 即当前时刻的加噪状态 $x_k$ 仅依赖于 上一时刻的状态 $x_{k-1}$, 与更早历史状态（如 $x_{k-2}, \dots, x_0$）无关.  $$q(x_k \mid x_{k-1}, \dots, x_0) = q(x_k \mid x_{k-1})$$
### 联合分布
从 $0$ 到 $T$ 的整个完整序列的联合分布如下 $$\begin{aligned}q(x_{0:T})&=q(x_0, x_1, \dots, x_T) \\&=p_\text{data}(x_0)\prod_{k=1}^Tq(x_k\mid x_{k-1})\end{aligned}$$
### 单步跳跃公式

> 由于高斯概率分布的性质, 多步加噪过程可以简化为一个单步条约公式

- 条件概率公式$$q(x_k​\mid x_0​)=\mathcal N(x_k​;\sqrt {\bar α_k}x_0​,(1−\bar α_k​)I)$$
- 单步跳跃公式 $$x_k = \sqrt {\bar α_k}​ x_0 + \sqrt{1−\bar α_k​}\ \epsilon, \qquad \epsilon\sim\mathcal N(0, I)$$
	- 其中 $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$
	- 当 $k\to\infty$时, $\bar\alpha_T​→0$, 有 $$x_T​≈N(0,I)$$
- **证明:**
	连续使用两次单步加噪公式, 有 $$\begin{aligned}x_t &= \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}} \epsilon_{t-2} \right) + \sqrt{1-\alpha_t} \epsilon_{t-1}\\&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \left( \sqrt{\alpha_t (1-\alpha_{t-1})} \epsilon_{t-2} + \sqrt{1-\alpha_t} \epsilon_{t-1} \right)\end{aligned}$$由于 $\epsilon_{t-1}$ 与 $\epsilon_{t-2}$ 为高斯分布, 他们的和依然为高斯分布, 其方差为 $\epsilon_{t-1}$ 与 $\epsilon_{t-2}$ 的方差之和$$\begin{aligned}\mathrm{Var}{(\bar{\epsilon}_{t-2})}&=\mathrm{Var}{(\epsilon_{t-1}}) +  \mathrm{Var}{(\epsilon_{t-2}}) \\&=\alpha_t (1-\alpha_{t-1}) + (1-\alpha_t)\\&= 1 - \alpha_t \alpha_{t-1}\end{aligned}$$合并后的加噪公式变为$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\epsilon}_{t-2}$$递归推广到$x_0$, 有$$x_t = \sqrt{\alpha_t \alpha_{t-1} \dots \alpha_1} x_0 + \sqrt{1 - \alpha_t \alpha_{t-1} \dots \alpha_1} \epsilon$$令 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$, 有$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$
# 反向去噪

> 我们希望学习到一个去噪核 $p_\theta(x_{k-1}\mid x_k)$ 尽量逼近真实反向核 $q(x_{k-1}\mid x_k)$

## 理想目标
### 贝叶斯公式
$$q(x_{k-1} \mid x_k) = \frac{q(x_k \mid x_{k-1}) q(x_{k-1})}{q(x_k)}$$
其中:
- $q(x_k​\mid x_k−1​)$ 是 **Gaussian**
- $q(x_{k−1}​)$ 是**边缘分布**，一般不知道显式形式

所以 $q(x_{k−1}​∣x_k​)$ 通常没有闭式公式    

### 小步长近似

当步长很小时，即前向每步扰动 βk 很小，这时可以证明或近似认为 $$q(x_{k−1}​∣x_k​)≈\mathrm{Gaussian}$$
#### 证明
> 利用Taylor 1st approximation来证明.

1. 由 Bayes 公式：$$q(x_{k-1} \mid x_k) = \frac{q(x_k \mid x_{k-1}) q(x_{k-1})}{q(x_k)}$$对两边取对数：$$\log q(x_{k-1} \mid x_k) = \log q(x_k \mid x_{k-1}) + \log q(x_{k-1}) - \log q(x_k)$$
2. 对于前向转移核（已知为高斯）：$$q(x_k \mid x_{k-1}) = \mathcal{N}(x_k; \sqrt{1-\beta_k}x_{k-1}, \beta_k \mathbf{I})$$当 $\beta_k \ll 1$ 时，$\sqrt{1-\beta_k} \approx 1$，其概率密度近似为：$$q(x_k \mid x_{k-1}) \propto \exp\left(-\frac{\|x_k - x_{k-1}\|^2}{2\beta_k}\right)$$因此：
$$\log q(x_k \mid x_{k-1}) = -\frac{\|x_k - x_{k-1}\|^2}{2\beta_k} + C_1$$

3. 将 $\log q(x_{k-1})$ 在 $x_k$ 处进行一阶 Taylor 展开：$$\begin{aligned}\log q(x_{k-1}) \approx &\log q(x_k) + (x_{k-1} - x_k)^\top \nabla_x \log q(x_k) \end{aligned}$$
4. 代入并忽略与 $x_{k-1}$ 无关的常数项：
$$\begin{align*}
\log q(x_{k-1} \mid x_k) &\approx -\frac{\|x_{k-1} - x_k\|^2}{2\beta_k} + (x_{k-1} - x_k)^\top \nabla_x \log q(x_k) + C \\
&= -\frac{\|x_{k-1} - x_k\|^2 - 2\beta_k(x_{k-1} - x_k)^\top \nabla_x \log q(x_k)}{2\beta_k} + C
\end{align*}$$
5. 对 $x_{k-1}$ 配平方并忽略与 $x_{k-1}$ 无关的常数项：$$\log q(x_{k-1} \mid x_k) = -\frac{\|x_{k-1} - [x_k + \beta_k \nabla_x \log q(x_k)]\|^2}{2\beta_k} + C$$因此得到：$$q(x_{k-1} \mid x_k) \approx \mathcal{N}\left(x_{k-1};\ x_k + \beta_k \nabla_x \log q(x_k),\ \beta_k \mathbf{I}\right)$$
#### 结论
- 当步长 $\beta_k$ 很小时，反向条件分布近似为高斯分布，因此我们可以考虑用一个高斯分布来拟合反向去噪过程
- 均值沿 score 方向修正为 $x_k + \beta_k \nabla_x \log q(x_k)$，方差为 $\beta_k \mathbf{I}$

### 引入初始条件
> 虽然 $q(x_{k-1} \mid x_k)$ 不可解，但考虑初始未加噪图像的 $q(x_{k-1} \mid x_k, x_0)$ 是**精确可解的 Gaussian**

#### Gaussian分布
根据 Bayes 定理展开：$$q(x_{k-1} \mid x_k, x_0) \propto q(x_k \mid x_{k-1}, x_0) q(x_{k-1} \mid x_0)$$而在 Markov 假设下，当前状态只依赖上一步，与更早的 $x_0$ 无关，所以：$$q(x_k \mid x_{k-1}, x_0) = q(x_k \mid x_{k-1})$$故而有$$q(x_{k-1} \mid x_k, x_0) \propto q(x_k \mid x_{k-1}) q(x_{k-1} \mid x_0)$$
此时后验概率为两个高斯分布相乘, 其依然为高斯分布

#### 均值与方差

已知前向过程的分布为：
- $q(x_k \mid x_{k-1}) = \mathcal{N}(\sqrt{\alpha_k}x_{k-1}, (1-\alpha_k) I)$
- $q(x_{k-1} \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{k-1}}x_0, (1-\bar{\alpha}_{k-1})I)$
- $q(x_k \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_k}x_0, (1-\bar{\alpha}_k)I)$

用 Bayes 公式组合：$$q(x_{k-1} \mid x_k, x_0) = \frac{q(x_k \mid x_{k-1})q(x_{k-1} \mid x_0)}{q(x_k \mid x_0)}$$可以算出均值和方差为
$$\begin{cases}&\tilde{\mu}_k(x_{k-1}|x_k, x_0) = \cfrac{\sqrt{\bar{\alpha}_{k-1}}(1-\alpha_k)}{1-\bar{\alpha}_k}x_0 + \cfrac{\sqrt{\alpha_k}(1-\bar{\alpha}_{k-1})}{1-\bar{\alpha}_k}x_k\\\\&\tilde{\beta}_k(x_{k-1}|x_k, x_0) = \cfrac{1-\bar{\alpha}_{k-1}}{1-\bar{\alpha}_k}(1-\alpha_k)\end{cases}$$
#### 模型拟合

我们希望使真实分布 $q(x_{k-1} \mid x_k, x_0) = \mathcal{N}(\tilde{\mu}_k, \tilde{\beta}_k I)$ 与模型拟合的分布 $p_\theta(x_{k-1} \mid x_k) = \mathcal{N}(\mu_\theta(x_k, k), \Sigma_\theta(x_k, k))$ 越接近越好, 即最小化 $q$ 与 $p_\theta$ 之间的 [[KL-Divergence]]$$KL[]$$





## 相关概念 / 前置知识 (Related Concepts / Prerequisites)

- [[相关概念A]]
- [[相关概念B]]
- [[相关概念C]]

## 关联资源 (Related Resources)

- **教材/讲义锚点：**
    - `[[教材或讲义名称]]`（章节/页码）
- **论文或文献：**
    - `[[文献笔记或论文标题]]`
- **补充资料：**
    - [外部链接标题](https://example.com)

# 未决问题 (Open Questions) [可选]
