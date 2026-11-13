# Taiji-Single-Sphere (V24)

## 简介

Taiji-Single-Sphere (V24) 是一个基于复数相角旋转与物理力场边界的序列建模架构。该模型去除了传统 Transformer 中的多头注意力机制（Multi-Head Attention）与显式 Softmax 概率分配，尝试通过显式的几何与动力学计算，完成序列信息的推演与特征提取。

通过引入视差力场（V20）、宏观相角计算与复数状态空间模型（Complex SSM），V24 能够在保持较低参数量的前提下，完成语义分类与物理规律预测，并提供内部计算过程的可解释性诊断视图。

## 核心架构

V24 引擎精简了原生 Mamba 的工程组件（如 1D 卷积、门控层），保留了最基础的状态空间递推骨架，并进行了复数域与相角驱动的扩展。

### 1. 视差力场边界（替代注意力分配）
模型在计算历史与当前状态的关系时，不依赖 Softmax 概率，而是通过计算向量方向的内积（余弦相似度）并施加因果掩码来实现：

\[
 S_{ij} = \frac{Z_i}{\|Z_i\|} \cdot \left( \frac{Z_j}{\|Z_j\|} \right)^T 
\]

随后计算一阶平滑力与三阶锐化力：

\[
 F_{macro} = \sum S_{ij}(Z_j - Z_i) 
\]
\[
 F_{micro} = \sum S_{ij}^3(Z_j - Z_i) 
\]

受力后的状态位移与能量约束公式为：

\[
 Z_{evolution} = Z_i + \frac{F_{macro} + F_{micro}}{\|Z_i\| + \epsilon} 
\]

最终将位移后的状态缩放回原模长，得到被“遮挡”后的视野剩余量 \(V_{obs}\)。

### 2. 宏观相角计算（前置 MLP）
前置 MLP 接收当前输入与视野剩余量，输出一个全局宏观相角 \(\theta_{global}\)，作为当前步的语境观测视角：

\[
 \theta_{global} = \pi \cdot \tanh(\text{MLP}([Z_i; V_{obs}])) 
\]

### 3. 复数相角干涉与状态演化（Mamba SSM）
为实现各特征分量的差异化演化，模型引入了可学习的频率参数 \(\omega_d\)，将宏观角度分配至各维度：

\[
 \theta_d = \theta_{global} \odot \omega_d 
\]

在状态递推过程中，利用欧拉公式对历史状态进行复数旋转：

\[
 h_{rot}^{real} = h_{t-1}^{real} \cos\theta_d - h_{t-1}^{imag} \sin\theta_d 
\]
\[
 h_{rot}^{imag} = h_{t-1}^{real} \sin\theta_d + h_{t-1}^{imag} \cos\theta_d 
\]

随后通过状态空间方程进行状态更新：

\[
 h_t = A \cdot h_{rot} + B \cdot x_t 
\]

### 4. 非线性对齐与输出
在物理演化完成后，通过后置 MLP 进行局部特征微调：

\[
 Z_{final} = h_t^{real} + \text{MLP}_{align}(h_t^{real}) 
\]

最终通过线性分类层（若为分类任务）或词表映射层进行输出。

## 实验验证

在单层网络结构的配置下（Transformer 保留 4 头注意力），V24 在三个测试任务上的表现如下：

*   **SST-2 情感分类**（自然语言因果）：Transformer 准确率为 67.09%，V24 准确率为 67.55%。
*   **残缺正弦波恢复**（含 30% 数据缺失的物理周期任务）：Transformer 准确率为 99.20%，V24 准确率为 90.00%。
*   **随机序列拷贝**（无规律纯随机任务）：Transformer 准确率为 15.60%，V24 准确率为 11.00%。

实验结果表明，在包含明确规律的数据集上，V24 架构能够提取有效的特征映射；而在面对无规律的随机数据时，基于物理推演的模型表现受限。同时，模型在运行过程中会生成包含相角轨迹、力场矩阵与复数波干涉的六维诊断图，可供分析其内部计算逻辑。

## 快速开始

### 依赖安装
```bash
pip install torch datasets transformers matplotlib numpy scikit-learn