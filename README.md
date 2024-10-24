# PAtt
This is the implementation for the paper [Probabilistic Attention for Sequential Recommendation](https://dl.acm.org/doi/10.1145/3637528.3671733) at KDD'24. 

Please cite our paper if you use the code:
```bibtex
@inproceedings{10.1145/3637528.3671733,
author = {Liu, Yuli and Walder, Christian and Xie, Lexing and Liu, Yiqun},
title = {Probabilistic Attention for Sequential Recommendation},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671733},
doi = {10.1145/3637528.3671733},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {1956–1967},
numpages = {12},
keywords = {attention mechanism, dpps, sequential recommendation},
location = {Barcelona, Spain},
series = {KDD '24}
}
```


## Paper Abstract
Sequential Recommendation (SR) navigates users' dynamic preferences through modeling their historical interactions. The incorporation of the popular Transformer framework, which captures long relationships through pairwise dot products, has notably benefited SR. However, prevailing research in this domain faces three significant challenges: (i) Existing studies directly adopt the primary component of Transformer (i.e., the self-attention mechanism), without a clear explanation or tailored definition for its specific role in SR; (ii) The predominant focus on pairwise computations overlooks the global context or relative prevalence of item pairs within the overall sequence; (iii) Transformer primarily pursues relevance-dominated relationships, neglecting another essential objective in recommendation, i.e., diversity. In response, this work introduces a fresh perspective to elucidate the attention mechanism in SR. Here, attention is defined as dependency interactions among items, quantitatively determined under a global probabilistic model by observing the probabilities of corresponding item subsets. This viewpoint offers a precise and context-specific definition of attention, leading to the design of a distinctive attention mechanism tailored for SR. Specifically, we transmute the well-formulated global, repulsive interactions in Determinantal Point Processes (DPPs) to effectively model dependency interactions. Guided by the repulsive interactions, a theoretically and practically feasible DPP kernel is designed, enabling our attention mechanism to directly consider category/topic distribution for enhancing diversity. Consequently, the <u>P</u>robabilistic <u>Att</u>ention mechanism (PAtt) for sequential recommendation is developed. Experimental results demonstrate the excellent scalability and adaptability of our attention mechanism, which significantly improves recommendation performance in terms of both relevance and diversity.

## Code introduction
The code is implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec) and [STOSA](https://github.com/zfan20/STOSA).


### Beauty dataset training and prediction
python main.py --model_name=DPPSAModel --data_name=beauty --output_dir=outputs/ --lr=0.001 --hidden_size=64 --max_seq_length=10 --hidden_dropout_prob=0.4 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --attention_probs_dropout_prob=0.0 --pvn_weight=0.005 --epochs=200 --gpu_id=0



