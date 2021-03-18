%% Full title
\title{Machine Learning 2 project: breast cancer cell classification through logistic regression and neural networks}

%% Shorter title, for use in page header
\runningtitle{ML2 project: breast cancer}
\author{Silke Meiner, Rafaela Neff}

# Abstract 
We present and compare machine learnt classification algorithms for diagnosing breast tumor cells as benign or malignant. The algorithms were trained on tabular data consisting of features extracted from microscopic images of fine-needle aspirates / biopsies.

We get reasonably good results from Logistic regression and were able to improve these results through single and multi-layer neural networks. The final neural network with an accuracy of ...,  sensitivity of ... and specificity ... is being audited at the German Gesundheitsministerium to become a state approved diagnostic tool.

# Business Understanding

Breast cancer is the most common cancer for women and ranks highest for cancer-related deaths in women in Germany: In 2016 there were 68,950 women and 710 men suffering from Breast Cancer (ICD-10 C50). In 2020 18,570 women and 1 men have died of breast cancer. \\

The situation is similar in many other countries.\\

Tumors can build in the human body as some cells grow more than they normally should. If this growth is not limiting itself and destroys body tissue and hinders body functions the tumor is labeled malignant and called cancer. \\

Tumors are classified in a binary fashion as either malignant or benign. Their difference in microscopic imagery are shown in figure \ref{fig:normal-vs-cancer}.
A typical first step when diagnosing a tumor is to do a fine needle aspirate (FNA) of a breast mass and looking at the cells through a microscope, describing characteristics of the cell nuclei. Further treatment differs according to the tumor diagnosis as benign or malignant.\\

\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{images/normal-vs-cancerous-cells.png}
    \caption{benign / normal and malignant / cancerous  cells}
    \label{fig:normal-vs-cancer}
\end{figure}

Sources : 
* https://www.krebsdaten.de/Krebs/EN/Content/Cancer\_sites/Breast\_cancer/breast\_cancer\_node.html \\

* https://www.verywellhealth.com/what-does-malignant-and-benign-mean-514240\\

* https://www.verywellhealth.com/cancer-cells-vs-normal-cells-2248794 \\


# Data sources and Data Understanding

The data for this project was collected in 1995 by the University of Wisconsin and made available to us through Prof. Dr. Nick Street of the University of Iowa. 

The data can be downloaded from the University of California, Irvine, [ https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29 ]

## Data Understanding

Our data set is mid sized with 569 observations for 30 numeric feature variables. Each observation has an additional binary diagnosis as benign or malignant. The data set is slightly unbalanced with 357 (63\%) being benign and 212 (37\%) malignant cases. We made malignant the positive class.\\

Some predictor variables are highly correlated.\\

When there is correlation of feature variables  with the target variable, it is mostly positive and not exceeding ...

## Data Visualisation, looped back after first modelling

This visualisation of our data ba application of PCA  was developed after a first loop in the CRISP model. In sequential reading this serves as a general visualisation now and will become important later on. 

** put images A.png and B.png here **

# Data Preparation

Since there were no missing data in our set we did not impute anything. Data was used as delivered.

Data was separated into training and test sets, each with a separate file. The data was split 80/20 and with stratification wrt to the diagnosis.\\

For further details, please look into our notebook on data preparation.

# Modeling
To solve the classification task we applied two machine learning methods: Logistic regression and neural networks.\\

In classification tasks we generally have predictor variables and a target variable. In binary classification the target variable can take one of two distinct values, interpreted as the two classes on option.\\

The methodological (?) similarities of both machine learning models are in the training and evaluation of the model. The training requires a training set of observed data points including the true values of the target variable. For evaluation a test set of observed  data points including the true values of the target variable is required. Training and test set need to be disjoint. On the test set the algorithm predicts classes for the data points and predictions are compared with the targets. Counting correct and not correct classifications and setting them in relation results in accuracy, sensitivity and specificity as measures of success.

## Logistic Regression

Given: Some numeric data, in tabular form of \(n\) rows and \(p+1\) variables. \(p\) variables are predictors and the remaining variable is the target. The target takes one of two values, interpreted as two classes, with one class defined as the positive class.\\

Desired: The class the data point belongs to with a probability distribution over the two classes (stating the probabilities that the data point belongs to each class).\\

Linear regression uses a linear combination of the variables to predict another numeric target variable. logistic regression does linear regression for the log-odds of the desired probabilities.
\begin{eqnarray*}
\text{log-odds} &=& \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p     
\end{eqnarray*}

The log-odds are transformed into (conditional) probabilities for the positive class through the logistic function \( \sigma(.) \), sometimes called sigmoid.

\begin{eqnarray*}
p(\bf{x}) = \sigma(\text{log-odds}) &=& \sigma(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p ) 
\end{eqnarray*}

with \( \bf{x}= ( x_1, \dots, x_p) \), \( p(\bf{x}) = \mathbf{P}\left(\text{target = positive class} | \text{predictors } = \bf{x} \right) \) and  \( \sigma(a) = \frac{1}{1+e^{-a}} \).\\

If the probability for the positive class \( p(\bf{x})\) exceeds a threshold, the data point is classified as the positive class.\\

The performance of the model is determined by its coefficients / weights \( \beta \). Finding the best / suitable coefficients is done in training. For logistic regression there are several training methods performed by statistical software like R.

In the beginning of running a logistic regression we ran into a 'problem' hinting to perfect separation of classes in our data set. We decided to ignore the problem and try to get the best possible results from logistic regression. Another option would have been: change to support vector machines which could directly exploit the seperability in our data.\\

Sources:\\
Log Regression https://christophm.github.io/interpretable-ml-book/logistic.html \\

for ignoring the problem of complete separation https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logistic-regression-and-what-are-some-strategies-to-deal-with-the-issue/ \\

Finding coefficients for logistic regression: https://stats.stackexchange.com/questions/344309/why-using-newtons-method-for-logistic-regression-optimization-is-called-iterati

## Neural Networks
Explain this, too.

# Final Assessment

We compare the logistic regression and the neural networks wrt accuracy and sensitivity and specificity on the test set.

\begin{center}
 \begin{tabular}{|c | c |  c | c|} 
 \hline
 Model & Accuracy & Sensitivity & Specificity \\ [0.5ex] 
 \hline\hline
 Logistic regression & .95 & .95 & .95 \\ 
 \hline
 Neural Network & .98 & .98 & .98 \\
 \hline
\end{tabular}
\end{center}

The neural network performs better in every criterion.\\

We should show AUC for both methods?!\\

please, see the notebook for more details.

# Deployment
Something on meeting the standards to deploy an algorithm in medical diagnosis. Who decides? What are the criteria.