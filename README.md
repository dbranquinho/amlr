## `AMLR` - Auto Machine Learning Report

Create a bealtifull Machine Learning Report with *`One-Line-Code`*

<hr>

![](https://img.shields.io/badge/pypi-1.4.1-blue) ![](https://img.shields.io/badge/python-3.7|3.8|3.9-lightblue) ![](https://img.shields.io/badge/Licence-MIT-lightgray) ![](https://img.shields.io/badge/status-Release-darkgreen) ![](https://img.shields.io/badge/pipeline-passed-green) ![](https://img.shields.io/badge/testing-passing-green) ![](https://img.shields.io/badge/H2O-Java-brown)


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
        - SOA1(Landis & Koch)	
        - SOA1(Fleiss)	
        - SOA1(Altman)	
        - SOA1(Cicchetti)	
        - SOA1(Cramer)	
        - SOA1(Matthews)	
        - TNR Macro	
        - TPR Macro	
        - FPR Macro	
        - FNR Macro	
        - PPV Macro	
        - ACC Macro	
        - F1 Macro	
        - TNR Micro	
        - FPR Micro	
        - TPR Micro	
        - FNR Micro	
        - PPV Micro	
        - F1 Micro	
        - Scott PI	
        - Gwet AC1	
        - Bennett S	
        - Kappa Standard Error	
        - Kappa 1% CI	
        - Chi-Squared	
        - Phi-Squared	
        - Cramer V	
        - Chi-Squared DF	
        - 1% CI	
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
rp.create_report(dataset='data/titanic-passengers.csv', target='Survived', max_runtime_secs=1)

webbrowser.open('report/index.html')
```
Another option is to load your own data set with `pandas` and switch, or parse, to the` AMLR` report command, but you cannot use both methods. The code will be:

```python
df = pd.read_csv('data/titanic-passengers.csv', sep=';')
rp.create_report(data_frame=df, target='Survived', max_runtime_secs=1)
```

### Parameters

* dataset: File to read by AMLR
* data_frame: Pandas DataFrame
* target: The target column
* duplicated: Default `True` Looking for duplicated lines
* sep: Default `;` if file is a csv, you must explicity the column sepatator character
* exclude: Default `True` a list with the columns to exclude to the process
* max_runtime_secs: Default `1` time limit to run deep learnig models

### max_run_time

When building a model, this option specifes the maximum runtime in seconds that you want to allot in order to complete the model. If this maximum runtime is exceeded before the model build is completed, then the model will fail.

Specifying max_runtime_secs=1 disables this option for production enviroment, thus allowing for an unlimited amount of runtime. If you just want to do a test, regardless of the results, use 1 seconds or a maximum of 61 seconds.

<BR>
<hr>
<BR>

## We tested with the following Dataset

- Classic dataset on `Titanic` disaster
    - Bernoulli Distribution Target or Binary Classification
    - Download here: [Titanic](https://public.opendatasoft.com/explore/dataset/titanic-passengers/table/)


### Output

See the output [here](https://www.thescientist.com.br/amlr/)

This is an example of the test made with the Titanic Dataset;

<hr>
<BR>

`enjoi!`
