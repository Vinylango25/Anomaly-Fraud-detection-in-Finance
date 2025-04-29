# Anomaly and Fraud Detection in Finance Using SHAP Explainers

## P33 Global Data Lab Weekly Report  
**Kipkemoi Vincent**  
**July 26, 2024**

---

### Objective

In this work, I have used two datasets: credit data and census data to evaluate and compare the computational efficiency (execution time) of 3 tree explainer algorithms:

- TreeShap algorithm built within the SHAP package  
- FastTreeShap v1 and FastTreeShap v2 algorithms built within the FastTreeShap package as defined in Lundberg, S. M., & Lee, S. I. (2017). The two algorithms are modifications of TreeShap to fully allow parallel computing.

From Lundberg, S. M., & Lee, S. I. (2017), the time complexity of a tree detector in calculating SHAP values is a function of a number of variables including the:

1. Number of samples used  
2. Number of estimators  
3. Maximum depth of each tree

Using Random Forest (RF) and Isolation Forest (IF) detectors and varying these variables, I have examined their execution times when calculating SHAP values using TreeShap, FastTreeShap v1 and FastTreeShap v2 algorithms.

Table 1 shows the description of the datasets. The number of instances in the credit transaction datasets was higher than 200,000. To make the SHAP calculations tractable in a reasonable time, I created a sub-sample of 100,000 instances for this dataset.

#### Table 1: Dataset description

| Dataset | # Instances         | # Attributes (original) | # Attributes (feature engineering) |
|---------|---------------------|--------------------------|-------------------------------------|
| Credit  | 100,000 (sub-sample)| 30                       | 34                                  |
| Census  | 48,882              | 14                       | 64                                  |

Prior to comparing the three SHAP algorithms in terms of execution times, we compared the SHAP values calculated by FastTreeShap v1 and FastTreeShap v2 against those calculated by the baseline algorithm (TreeShap), and found that the maximum difference is insignificant (lower than 10⁻⁷).

Figure 1 shows the top 3 feature rankings based on SHAP values for RF model implemented on Census data. The results show that the SHAP calculations based on TreeShap, FastTreeShap v1 and FastTreeShap v2 algorithms are similar and provide similar explainability.

![Figure 1: RF SHAP calculations on census data obtained using the three algorithms](f1.png)

> Data Sources:  
> - [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
> - [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income)

---

### Varying the Number of Samples

Figure 2 shows the execution times for each of the SHAP algorithms when the number of samples is varied from 1,000 to 10,000. Note that to effect these calculations, the number of estimators is set to 100 for both models. The maximum depth parameter is set to 8 for the RF model while for IF, based on its documentation, it is already fixed at ceil(log₂(n)), where *n* is the number of samples.

![Figure 2: Running times for varying number of samples](f2.png)

The results show that SHAP calculations via FastTreeShap v1 and FastTreeShap v2 are generally faster compared to TreeShap. The speed-up in SHAP calculations is more pronounced for FastTreeShap v2 compared to FastTreeShap v1, especially at higher sample sizes.

---

### Varying the Maximum Depth

As initially mentioned, the maximum depth of IF is fixed at ceil(log₂(n)). Here I only examined the effect of RF’s maximum depth in SHAP calculation execution for the two datasets. To do this, I have fixed the number of samples at 10,000 and set the RF number of estimators at 100.

![Figure 3: Running times for varying maximum depth](f3.png)

The result in Figure 3 shows that at lower values of maximum depth (<4), the difference in the execution times for the three algorithms seems to be minimal. As the values of maximum depth increase, the SHAP calculation execution times generally grow exponentially. The speed-up for FastTreeShap v2 is more pronounced than that of FastTreeShap v1, especially at higher values of maximum depth.

---

### Varying the Number of Estimators

To assess the effect of a model’s number of estimators on SHAP calculation execution time, I fixed the number of samples at 10,000 for the two models and maximum depth at 8 for the RF model.

Figure 4 shows the execution time of the three algorithms when the number of estimators for each model is varied from 40 to 200. It is worth noting that the speed-up for FastTreeShap v2 is significantly higher compared to that of FastTreeShap v1, especially when the number of estimators is set at high values.

At smaller values of number of estimators (<60), SHAP calculations using TreeShap seem to be generally faster than calculations using FastTreeShap v1. As the number of estimators increases beyond 60, however, the execution time for FastTreeShap v1 improves compared to TreeShap.

![Figure 4: Running times for varying number of estimators](f4.png)

---

### Summary

Table 2 illustrates the average speed-up achieved by FastTreeShap v1 and FastTreeShap v2 in computing SHAP values for Credit and Census data. Depending on the dataset and the variable of concern, the average speed-up for FastTreeShap v1 ranges from 1.08 to 1.84, while that of FastTreeShap v2 ranges from 1.67 to 4.58.

#### Table 2: Average Speed-Up

| Dataset | Variable         | Speed-up (RF)         | Speed-up (IF)         |
|---------|------------------|------------------------|------------------------|
|         |                  | v1       | v2          | v1       | v2          |
| Credit  | No. of samples   | 1.84×    | **4.58×**   | 1.29×    | 3.15×       |
|         | Maximum depth    | 1.35×    | 3.03×       | –        | –           |
|         | No. of estimators| 1.13×    | 2.35×       | 1.29×    | 3.15×       |
| Census  | No. of samples   | 1.12×    | 1.69×       | 1.16×    | 2.07×       |
|         | Maximum depth    | 1.15×    | 1.67×       | –        | –           |
|         | No. of estimators| 1.12×    | 2.01×       | 1.08×    | 2.03×       |

---

### Conclusion

The comparison of SHAP value computation algorithms revealed that **FastTreeShap v2** significantly accelerates execution times compared to TreeShap and FastTreeShap v1.

- Achieved up to a **4.58× speed-up** on the Credit dataset  
- Up to **2.07×** on the Census dataset  
- This improvement is most notable with increased sample sizes, deeper tree depths, and more estimators

These results underscore **FastTreeShap v2’s superior efficiency**, making it highly suitable for **large-scale and complex model analyses**. Overall, FastTreeShap v2 offers substantial computational advantages, enhancing practical applicability in machine learning tasks.

---

### References

- Lundberg, S. M., & Lee, S. I. (2017). *Fast TreeSHAP: Accelerating SHAP Value Computation for Trees*. Proceedings of the 34th International Conference on Machine Learning (ICML)  
- Molnar, C. (2019). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*  
- Sengupta. (2020). *Interpretable Machine Learning: Introduction to SHAP Values*. Towards Data Science

