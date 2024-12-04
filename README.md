# NeurIPS 2024: RAGraph: A General Retrieval-Augmented Graph Learning Framework

Welcome to the official GitHub repository for paper: **RAGraph: A General Retrieval-Augmented Graph Learning Framework** ([https://arxiv.org/abs/2408.09199](https://arxiv.org/abs/2410.23855))
![image](https://github.com/user-attachments/assets/53bca93b-0ca5-40ea-a70f-1bd49e26cf17)


## Overview

Graph Neural Networks (GNNs) have become essential in interpreting relational data across various domains, yet, they often struggle to generalize to unseen graph data that differs markedly from training instances. In this paper, we introduce a novel framework called General **R**etrieval-**A**ugmented **Graph** Learning (**RAGraph**), which brings external graph data into the general graph foundation model to improve model generalization on unseen scenarios. On the top of our framework is a toy graph vector library that we established, which captures key attributes, such as features and task-specific label information. During inference, the **RAGraph** adeptly retrieves similar toy graphs based on key similarities in downstream tasks, integrating the retrieved data to enrich the learning context via the message-passing prompting mechanism. Our extensive experimental evaluations demonstrate that **RAGraph** significantly outperforms state-of-the-art graph learning methods in multiple tasks such as node classification, link prediction, and graph classification across both dynamic and static datasets. Furthermore, extensive testing confirms that **RAGraph** consistently maintains high performance without the need for task-specific fine-tuning, highlighting its adaptability, robustness, and broad applicability.

![image](https://github.com/user-attachments/assets/4f6e5e70-4012-4f34-9435-cec3893c4d88)


## Install Environment

We use conda to manage the environment.
Please refer to the following steps to install the environment:

```sh
conda create -n TCRAG python=3.11 -y
conda activate TCRAG
pip install -r requirements.txt
python -m spacy download zh_core_web_trf
```

## Folder Structure

The important data structure are as follows:

```tex
└── code-and-data
    ├── data                    # Including datasets-CMB, Clin and MMCU
    ├── model                   # The core source code of our model TC-RAG and baselines
    │   |──  _init_.py          # Initialization file for models
    │   |──  system_score.py    # Including score computation code in TC-RAG   
    │   |──  other-model.py     # Including the base model and other RAG methods 
    ├── microservice            # Defination the microservice code
    │   |──  _init_.py          # Initialization file for microservice
    │   |──  BingSearch.py      # BingSearch Tool, remember to register to get your API    
    │   |──  DocumentSearch.py  # DocumentSearch tool
    │   |──  Generator.py       # This is the TC-RAG-specific code for generate logits, attention, and entropy when inference...
    │   |──  CustomLanguageModel# Your Custom Language Model 
    │   |──  config.py          # Path for your local LLMs' paths and your LoRA weight
    ├── mian.py                 # This is the main file
    ├── requirements.txt        # The python environment needed for TC-RAG
    └── README.md               # This document
```


## Setup Basic Config for Large Language Model

TC-RAG mainly supports Large Language Models Qwen, which is a series of transformer-based large language models by Alibaba Cloud.

### Deploy a Large Language Model in Local

If you want to deploy a large language model in local, just change the `model_path` in `microservice/config.py` to actual path of your model. The variable `model_path` should be the path to the directory containing the model files. 
Besides, if you want to use a finetuned large language model with lora weights, you can set `lora_model_path` to the path of the directory containing the lora weight files.

### Use a Large Language Model in Cloud

Some baseline methods do not require treating the LLM as a whitebox system. Therefore, we provide a simple interface to use a LLM in cloud. Just chagne the URL defined in `microservice/CustomLanguageModel.py` to the URL of your own deployed LLM, or use dashscope to call the LLM in cloud. If you use dashscope, please set the `OPENAI_API_KEY` environment variable to your own API key in `.env` file.

## Running

To run the code, simply execute the following command:

```sh
python main.py
```

And we have provided some arguments to run the code with different baseline methods, different datasets, and different large language models. You can enable all these augments with the following command:

```sh
python main.py --module_name "your_module_name" --model_name "your_model_name" --dataset_name "your_dataset_name"
```

The following table lists all the available arguments, their default values and options for each argument:

| Argument            | Default Value | Options                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
| ------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| `--module_name`     | `Base`       | `Base`, `CoT`, `Sure`, `BasicRAG`, `TokenRAG`, `EntityRAG`, `SentenceRAG`, `TCRAG`
| `--model_name`      | `Qwen` | `Qwen`(used for local LLM), `Aliyun`(used for cloud LLM), `Xiaobei`(used for finetuned LLM)
| `--dataset_name`    | `CMB` | `CMB`, `MMCU`, `Clin`

## Contact
For any questions or suggestions, please contact (Rihong Qiu:rihongqiu@stu.pku.edu.cn) and (Xinke Jiang:xinkejiang@stu.pku.edu.cn).

## Citation
```tex
@@inproceedings{jiang2024ragraphgeneralretrievalaugmentedgraph,
      title={RAGraph: A General Retrieval-Augmented Graph Learning Framework}, 
      author={Xinke Jiang and Rihong Qiu and Yongxin Xu and Wentao Zhang and Yichen Zhu and Ruizhe Zhang and Yuchen Fang and Xu Chu and Junfeng Zhao and Yasha Wang},
      booktitle={NeurIPS},
      year={2024},
}
```
