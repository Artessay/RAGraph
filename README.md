# NeurIPS 2024: RAGraph: A General Retrieval-Augmented Graph Learning Framework

Welcome to the official GitHub repository for paper: **RAGraph: A General Retrieval-Augmented Graph Learning Framework** ([https://arxiv.org/abs/2408.09199](https://arxiv.org/abs/2410.23855))


![image](https://github.com/user-attachments/assets/53bca93b-0ca5-40ea-a70f-1bd49e26cf17)



## Overview

Graph Neural Networks (GNNs) have become essential in interpreting relational data across various domains, yet, they often struggle to generalize to unseen graph data that differs markedly from training instances. In this paper, we introduce a novel framework called General **R**etrieval-**A**ugmented **Graph** Learning (**RAGraph**), which brings external graph data into the general graph foundation model to improve model generalization on unseen scenarios. On the top of our framework is a toy graph vector library that we established, which captures key attributes, such as features and task-specific label information. During inference, the **RAGraph** adeptly retrieves similar toy graphs based on key similarities in downstream tasks, integrating the retrieved data to enrich the learning context via the message-passing prompting mechanism. Our extensive experimental evaluations demonstrate that **RAGraph** significantly outperforms state-of-the-art graph learning methods in multiple tasks such as node classification, link prediction, and graph classification across both dynamic and static datasets. Furthermore, extensive testing confirms that **RAGraph** consistently maintains high performance without the need for task-specific fine-tuning, highlighting its adaptability, robustness, and broad applicability.


![image](https://github.com/user-attachments/assets/4f6e5e70-4012-4f34-9435-cec3893c4d88)


## Contact
For any questions or suggestions, please contact [Rihong Qiu](rihongqiu@stu.pku.edu.cn) and [Xinke Jiang](xinkejiang@stu.pku.edu.cn).

## Citation
```tex
@@inproceedings{jiang2024ragraphgeneralretrievalaugmentedgraph,
      title={RAGraph: A General Retrieval-Augmented Graph Learning Framework}, 
      author={Xinke Jiang and Rihong Qiu and Yongxin Xu and Wentao Zhang and Yichen Zhu and Ruizhe Zhang and Yuchen Fang and Xu Chu and Junfeng Zhao and Yasha Wang},
      booktitle={NeurIPS},
      year={2024},
}
```
