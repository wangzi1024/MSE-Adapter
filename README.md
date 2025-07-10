# 基于MSE-Adapter的多模态情感分析复现与改良

本项目是对论文《MSE-Adapter: A Lightweight Plugin Endowing LLMs with the Capability to Perform Multimodal Sentiment Analysis and Emotion Recognition》的深度复现与探索性改良。我们不仅验证了原论文的核心结论，还在此基础上，通过引入新的模块化设计，对原模型进行了两阶段的迭代优化，并最终取得了性能上的提升。

## 项目结构

本代码库包含三个核心分支，分别对应我们研究工作的不同阶段：

* **`MSE-Qwen-1.8B/`**: 此目录包含了对原论文**MSE-Adapter**的忠实复现代码。我们以Qwen-1.8B为骨干网络，在MELD和SIMS-V2数据集上验证了其有效性。

* **`EBlock-MSE-Qwen-1.8B/`**: 这是我们第一阶段的改良尝试。该版本将原模型中处理视频特征的sLSTM模块，替换为了一个结合了空间注意力和频域分析的**EBlock**模块。

* **`Multi-GTU-MSE-Qwen-1.8B/`**: 这是我们最终成功的改良版本。在EBlock的基础上，我们进一步将原有的MSF融合模块，替换为了一个基于多尺度门控卷积的**Multi-GTU**模块，实现了更强大的特征融合能力。
