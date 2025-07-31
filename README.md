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

## CODE

Main experiment pipeline is in ***normalized_segmentation.ipynb***

Other relevant notebooks are:

***evaluate_explanations.ipynb*** 

***evaluate_opposite_classes.ipynb***

***evaluate_order_change.ipynb***

***evaluate_segmentations.ipynb***

## DATA 

[Datasets](https://drive.google.com/file/d/19zQgX_w83H1kTwlB1dLuQT8p_bM_sk1B/view?usp=drive_link 'Datasets')

[Trained models](https://drive.google.com/drive/folders/1C1KBoaCV8ZmjusNXEr8itCsNfikhPVtq?usp=drive_link 'Trained models')

[Attributions](https://drive.google.com/file/d/17M9CLJfbnAOBScfSeUTnwYTss26Z3eT8/view?usp=drive_link 'Attributions')

[AUCDiff results](https://drive.google.com/drive/folders/1br4IIjng3kQIoRE_go2skrQA8R5nzoKB?usp=drive_link 'AUCDiff results') as CSV files

[InterpretTime result](https://drive.google.com/drive/folders/1DfUtAZN6DLn_6eas_Etx9dcxqROQnlKf?usp=drive_link 'InterpretTime results') as numpy files (.npy)

