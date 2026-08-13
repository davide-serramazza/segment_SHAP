# SEGMENT SHAP

Repository for "An Empirical Evaluation of Factors Affecting SHAP Explanation of Time Series Classification" 
accepted for TempXAI 2025 Workshop for Explainable AI in Time Series and DataStreams (ECML PKDD 2025)

## ABSTRACT 

Explainable AI (XAI) has become an increasingly important topic for understanding and attributing the predictions made 
by complex Time Series Classification (TSC) models.
Among attribution methods, SHapley Additive exPlanations (SHAP) is widely regarded as an excellent attribution method,
but its computational complexity, which scales exponentially with the number of features, limits its practicality for 
long time series.
To address this, recent studies have shown that aggregating features via segmentation, to compute a single attribution 
value for a group of consecutive time points, drastically reduces SHAP running time. However, the choice of the optimal 
segmentation strategy remains an open question.
In this work, we investigated eight different Time Series Segmentation algorithms to understand how segment compositions 
affect the explanation quality. We evaluate these approaches using two established XAI evaluation methodologies: 
InterpretTime and AUC Difference.
Through experiments on both Multivariate (MTS) and Univariate Time Series (UTS), we find that the number of segments has
a greater impact on explanation quality than the specific segmentation method. Notably, equal-length segmentation
consistently outperforms most of the custom time series segmentation algorithms. Furthermore, we introduce a novel 
attribution normalisation technique that weights segments by their length and we show that it consistently improves 
attribution quality. 

## IMAGES

![image](https://github.com/davide-serramazza/segment_SHAP/blob/main/evaluation/images/2%20factor%20aggregate/Aggregate%20Results%20for%20AUCD%20by%20ML%20model%20and%20Dataset.png)
Aggregate AUCD scores by ML models and datasets 

![image](https://github.com/davide-serramazza/segment_SHAP/blob/main/evaluation/images/2%20factor%20aggregate/Aggregate%20Results%20for%20AUCSE%20by%20ML%20model%20and%20Dataset.png)
Aggregate AUCSE scores by ML models and datasets 

## CODE

Main experiment pipeline is in ***normalized_segmentation.ipynb***

Other relevant notebooks are:

***evaluate_explanations.ipynb*** 

***evaluate_opposite_classes.ipynb***

***evaluate_order_change.ipynb***

***evaluate_segmentations.ipynb***

## DATA 

Datasets, trained models, computed attributions and results of XAI evaluations methods are available through [zenodo](https://zenodo.org/uploads/20608403)

## HOW TO CITE 

@InProceedings{10.1007/978-3-032-19105-2_32,
author="Papadeas, Nikos
and Serramazza, Davide Italo
and Abdallah, Zahraa
and Ifrim, Georgiana",
editor="Koprinska, Irena
and Mendes-Moreira, Jo{\~a}o
and Branco, Paula",
title="An Empirical Evaluation of Factors Affecting SHAP Explanation of Time Series Classification",
booktitle="Machine Learning and Principles and Practice of Knowledge Discovery in Databases",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="458--473",
isbn="978-3-032-19105-2"
}

