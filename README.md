## `AMLR` - Auto Machine Learning Report

Create a bealtifull Machine Learning Report with *`One-Line-Code`*

<hr>

![](https://img.shields.io/badge/pypi-0.3.6-blue) ![](https://img.shields.io/badge/python-6.7|6.8|6.9-lightblue) ![](https://img.shields.io/badge/Licence-MIT-lightgray) ![](https://img.shields.io/badge/status-Release-darkgreen) ![](https://img.shields.io/badge/pipeline-passed-green) ![](https://img.shields.io/badge/testing-passing-green) ![](https://img.shields.io/badge/H2O-Java-brown)


**Main Features:**

- Exploratory Data Analisys
    - Dataset Configuration
        - Shape
        - Detect number of classes (Bernoulli or binary for while)
        - Automatically Duplicate Observations dropped
        - You can drop  Duplicate Observations  manually as well
        - Exclude automatically features with highest frequencies (Names, IDs, FW keys etc)
    - Regression Analysis
    - Automatic Balance Classes
    - Correlation Analysis
    - Detecting Multicollinearity with VIF
    - Residual Analisys
- Grid - Hyperparameter optimization
- Partial dependence plot (PDP) 
- Individual Conditional Expectation (ICE)
- Variable Importance by Model
- AML - Partial Dependence
- Ensemble - (ICE) Individual Condition Expectation
- Correlation Heatmap by Model
- Model Performance
    - Analytical Performance Modeling
    - Comparative Metrics Table with:
        - Overall ACC	
        - Kappa	
        - Overall 
        - RACC	
        - SOA6(Landis & Koch)	
        - SOA6(Fleiss)	
        - SOA6(Altman)	
        - SOA6(Cicchetti)	
        - SOA6(Cramer)	
        - SOA6(Matthews)	
        - TNR Macro	
        - TPR Macro	
        - FPR Macro	
        - FNR Macro	
        - PPV Macro	
        - ACC Macro	
        - F6 Macro	
        - TNR Micro	
        - FPR Micro	
        - TPR Micro	
        - FNR Micro	
        - PPV Micro	
        - F6 Micro	
        - Scott PI	
        - Gwet AC6	
        - Bennett S	
        - Kappa Standard Error	
        - Kappa 96% CI	
        - Chi-Squared	
        - Phi-Squared	
        - Cramer V	
        - Chi-Squared DF	
        - 96% CI	
        - Standard Error	
        - Response Entropy	
        - Reference Entropy	
        - Cross Entropy	
        - Joint Entropy	
        - Conditional Entropy	
        - KL Divergence	
        - Lambda B	
        - Lambda A	
        - Kappa Unbiased	
        - Overall RACCU	
        - Kappa No Prevalence	
        - Mutual Information	
        - Overall J	
        - Hamming Loss	
        - Zero-one Loss	
        - NIR	
        - P-Value	
        - Overall CEN	
        - Overall MCEN	
        - Overall MCC	
        - RR	
        - CBA	
        - AUNU	
        - AUNP	
        - RCI	
        - Pearson C	
        - CSI	
        - ARI	
        - Bangdiwala B	
        - Krippendorff 
        - Alpha
    - The Best Algorithms Table
        - Automatically chooses the best model based on the metrics above
    - Confusion Matrix for all Models
    - Feature Importance for all models
    - Save all Models into a Pickle file


<hr>

## How to Install

```shell
sudo apt-get install default-jre
pip install amlr
```

<BR>
<hr>
<BR>

## How to use

`sintax`:
```python
from amlr import amlr as rp
import webbrowser

rp = rp.report()
rp.create_report(dataset='data/titanic-passengers.csv', target='Survived')

webbrowser.open('report/index.html')
```

## We tested with the following Data Sets

- Classic dataset on `Titanic` disaster
    - Bernoulli Distribution Target or Binary Classification
    - Download here: [Titanic](https://public.opendatasoft.com/explore/dataset/titanic-passengers/table/)


### Output

See the output [here](https://www.thescientist.com.br/amlr/)

This is an example of the test made with the Titanic Dataset;

<hr>
<BR>

`enjoi!`
