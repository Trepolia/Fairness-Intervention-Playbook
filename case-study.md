## Executive Summary

**Target audience:** ML engineers and data scientists implementing fairness improvements at mid-sized bank implementing a machine learning system for loan approval decisions.

**Purpose:**  To identify the underlying mechanisms creating unfairness, that will guide selection of appropriate interventions.

**Prerequisites:** Completed fairness audit identifying specific issues to address:
    At a mid-sized bank implementing a machine learning system for loan approval decisions. The system will analyze applicant data including credit history, income, debt-to-income ratio, employment history, and property value to predict default risk. Based on this risk assessment, the system will either approve or deny loan applications.
    A preliminary analysis revealed concerning patterns: the model approves loans for male applicants at significantly higher rates than for similarly qualified female applicants. The bank's data science team suspects this disparity stems from historical lending practices reflected in their training data.

# 1. Variable Identification:

    protected attributes (race, gender), outcomes (loan approval), mediators (credit score, income), and confounders (neighborhood characteristics, economic conditions). Relationships between these variables would be determined through domain expertise, existing research, and potentially causal discovery techniques applied to data.
    
# 2. Causal Graph (DAG)
```mermaid
graph LR
    %% Protected Attribute
    Gender[Gender]
    
    %% Mediators/Compounders
    EmpHist[Employment History]
    Income[Income Level]
    DTI[Debt-to-Income Ratio]
    IncStability[Income Stability]
    JobSecurity[Job Security]
    
    %% Proxies
    PTStatus[Part-time Status<br/>Proxy]
    Industry[Industry Sector<br/>Proxy]
    
    %% Intermediate Outcome
    DefaultRisk[Default Risk]
    
    %% Final Outcome
    Approval([Approval Decision<br/>Outcome])
    
    %% Causal pathways (solid arrows)
    Gender -->|causal| EmpHist
    Gender -->|causal| Income
    EmpHist -->|causal| DefaultRisk
    Income -->|causal| DTI
    DTI -->|causal| DefaultRisk
    IncStability -->|causal| DefaultRisk
    JobSecurity -->|causal| DefaultRisk
    DefaultRisk -->|causal| Approval
    
    %% Correlations (dashed arrows)
    Gender -.->|correlation| PTStatus
    Gender -.->|correlation| Industry
    PTStatus -->|causal| IncStability
    Industry -->|causal| JobSecurity
    
    %% Styling
    classDef protected fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    classDef mediator fill:#4dabf7,stroke:#1971c2,stroke-width:2px
    classDef proxy fill:#ffd43b,stroke:#fab005,stroke-width:2px
    classDef outcome fill:#69db7c,stroke:#2f9e44,stroke-width:3px
    
    class Gender protected
    class EmpHist,Income,DTI,IncStability,JobSecurity mediator
    class PTStatus,Industry proxy
    class DefaultRisk,Approval outcome

    ```

## 4 Evaluation Framework
This evaluation framework is designed for the bank’s loan approval system to ensure that fairness interventions—whether implemented at the data, model, or prediction level—are rigorously measured across three critical pillars: **Fairness Impact**, **Information Preservation**, and **Computational Cost**.

### **1. Fairness Impact Evaluation**
This pillar measures the success of the intervention in reducing disparities identified during the audit and causal analysis phases.

*   **Statistical Fairness Metrics:**
    *   **Demographic Parity Difference:** Measures the gap in approval rates between demographic groups.
    *   **Equal Opportunity Disparity:** Evaluates the difference in true positive rates (correctly approved qualified applicants) across groups.
    *   **Equalized Odds:** Assesses disparities in both true positive and false positive rates.
*   **Calibration & Reliability Metrics:**
    *   **Expected Calibration Error (ECE) and Maximum Calibration Error (MCE):** Quantify how well the predicted risk probabilities align with actual default frequencies for each demographic group.
    *   **Group-Specific Reliability Diagrams:** Visual assessment of whether a "70% risk" score represents the same actual risk across different demographic groups.
*   **Causal & Counterfactual Metrics:**
    *   **Counterfactual Effect Size:** Measures the average difference in loan predictions if an individual's protected attribute were changed while keeping independent factors constant.
    *   **Counterfactual Fairness Violation Rate:** The percentage of applicants for whom the model would change its decision solely based on a change in a protected attribute.
    *   **Path-Specific Effect Measures:** Isolate the impact of the protected attribute along specific "unfair" pathways (e.g., zip code) versus "fair" pathways (e.g., debt-to-income ratio).
*   **Intersectional Assessment:** Fairness metrics must be disaggregated to evaluate performance for overlapping identities (e.g., Black women) to ensure aggregate improvements do not mask subgroup disparities.

### **2. Information Preservation and Utility**
In a banking context, maintaining the predictive utility of the model is vital for risk management and profitability.

*   **Predictive Performance Metrics:** Standard measures include changes in **Accuracy**, **AUC-ROC**, **Precision**, and **F1-score** after the intervention.
*   **Fidelity and Information Loss:**
    *   **Rank-Ordering Preservation:** Especially in lending, it is critical that the intervention does not flip the relative risk ranking of applicants within a group.
    *   **Mutual Information Analysis:** Measures how much "useful" information is retained in transformed representations compared to original features.
*   **Trade-off Analysis:**
    *   **Pareto Frontier Exploration:** Plotting models to identify the "knee" where the maximum fairness gain is achieved with minimal accuracy loss.
    *   **"Price of Fairness":** A quantitative calculation of the performance reduction (e.g., a 1% drop in AUC) per unit of fairness improvement (e.g., a 10% reduction in approval gap).

### **3. Computational and Operational Cost**
This pillar assesses the feasibility of the intervention within the bank's production environment.

*   **Training and Development Costs:**
    *   **Training Time Increase:** Measures the additional overhead introduced by complex in-processing techniques like adversarial debiasing or constrained optimization, which can increase training time by 30-50%.
    *   **Hyperparameter Tuning Complexity:** The effort required to tune regularization weights ($\lambda$) or adversarial components.
*   **Inference and Deployment Costs:**
    *   **Inference Latency:** The additional time required to apply transformations or calibration lookups during real-time loan processing.
    *   **Memory and Storage Requirements:** The cost of maintaining separate calibration curves or large generative models for synthetic data.
*   **Operational Integration:** The effort required for regulatory review and the ability to integrate the intervention into legacy pipelines without a full system overhaul.

### **4. Robustness and Validation Framework**
To verify the effectiveness of the audit and intervention, the framework includes "quality control" steps:

*   **Sensitivity Analysis (E-values):** Quantifies how robust causal fairness conclusions are to potential unmeasured confounders, such as "quality of prior legal representation".
*   **Structural Validation:** Testing implied **conditional independencies** (d-separation) in the causal model against observational data to ensure the model reflects reality.
*   **Generalization Testing:** Verifying that fairness improvements hold on held-out test data and remain stable under synthetic distribution shifts.