# Causal Fairness Toolkit - Building causal models to detect Bias

## **Purpose & Scope**

- **Problem:** Detect and attribute discrimination that correlation-based metrics miss by modeling causal mechanisms behind outcomes.
- **Primary function:** Build DAGs/SEMs, run counterfactual queries, compute path-specific effects, and surface whether disparities arise from direct, indirect, or proxy pathways.
- **Outputs:** Causal graphs, structural equations, validation evidence, counterfactual metrics, intervention recommendations (pre/in/post-processing).
- Scope / limits: Works from observational data and domain assumptions; requires expert validation and cannot fully prove causation without strong instruments or experiments.

## **Primary Stakeholders**

- **Owner:** Model Risk / Model Governance team — maintains toolkit, sets standards, approves use.
- **Executors:** Data scientists, causal analysts, and ML engineers — build DAGs, fit SEMs, run counterfactual analyses, implement chosen interventions.
- **Validators:** Subject-matter experts and legal/compliance reviewers — assess causal assumptions, legitimacy of mediators, and regulatory fit.
- **Operators:** MLOps / Monitoring teams — integrate validation checks into CI/CD and runtime monitoring.
- **Beneficiaries:** Impacted communities, product teams, execs, and regulators — receive fairer decisions, documented risk assessments, and remediation actions.

## **Integration Points in the GRC Lifecycle**

- **Policy & Standards:** Feeds into model governance policies by defining acceptable causal assumptions, mediation rules, and documentation requirements.
- Risk Assessment / Model Inventory: Used during model intake and risk scoring to flag high-risk models and document causal-risk rationales.
- **Design & Review (Development):** Embedded in design reviews—DAGs and SEMs are required artifacts before training or deployment.
- **Validation & Testing:** Supplies statistical independence tests, sensitivity analyses (E-values), and counterfactual metrics for validation reports.
- Approval & Sign-off: Provides the technical evidence package for governance committees and legal sign-off.
- **Monitoring & Controls:** Produces monitoring metrics and thresholds for drift or new proxy pathways; triggers remediation workflows.
- **Audit & Compliance:** Generates reproducible documentation and evidence (graphs, equations, validation tables) for internal audits and regulator requests.

Traditional statistical parity metrics (e.g., demographic parity) often fail because they treat surface-level correlations as evidence of discrimination without modeling the data-generating process. A causal-first approach follows Pearl’s (2009) assertion that "behind every successful policy to correct discrimination lies a causal understanding." We categorize discrimination into three structural patterns detectable via **Directed Acyclic Graphs (DAGs)**:
- **Direct Discrimination**: Causal paths leading directly from protected attributes (A) to the outcome (Y).
- **Indirect Discrimination**: Paths where A influences Y through intermediate Mediators (M) which may or may not be legally/ethically permissible (e.g., A→M→Y).
- **Proxy Discrimination**: Relationships where a feature (X) serves as a surrogate for A due to a shared ancestor or historical correlation (e.g., Zip Code as a proxy for Race), even in the absence of a direct causal link to the outcome.


## Causal Modelling Process:

## 1 Identify and Classify Variables

<details>
   <summary>
   Guidelines
   </summary>
      Loftus et al. (2018) provide a framework for this process in their work on causal reasoning for algorithmic fairness. They demonstrate techniques for variable identification based on domain knowledge, data analysis, and fairness requirements. Their approach emphasizes the importance of including variables that might reveal indirect discrimination or proxy effects, not just those directly used in the model.
</details>

   **Variable Identification Template:**

   - Protected Attribute (A) Identification
     1. Primary protected attributes: [List legally protected characteristics relevant to this application]
     2. Intersectional categories: [List relevant combinations of protected attributes]

   - Outcome Variable (Y) Identification
     1. Decisions or predictions made by system: [List outcomes]
     2. Evaluation metrics used: [List metrics]

   - Mediator Variable (M) Identification
     1. Variables directly influenced by protected attributes: [List variables]
     2. Evidence for causal relationship: [Brief justification per variable]

   - Proxy Variable (P) Identification
     1. Variables correlated with protected attributes: [List variables]
     2. Evidence for correlation: [Brief justification per variable]
     3. Common causes explaining correlation: [Explanation per variable]

   - Confounders (C): 
     1. External factors affecting both (A) and (Y): [List variables]
     2. Document your reasoning

   - Legitimate Predictor Identification
     1.  Variables that should influence outcomes: [List variables]
     2.  Justification for legitimacy: [Brief justification per variable]


## 2 Construct Causal Graph (DAG)

**Causal Graph Construction Guidelines:**

* [ ]Use directed arrows to represent causal relationships.
* [ ]Use bidirectional dashed arrows to represent correlations without direct causation.
* [ ]Distinguish protected attributes, mediators, proxies, and outcomes using different node shapes.
* [ ] **Temporal Ordering Check:** Does every cause in the graph precede its effect in time?
* [ ] **Intersectional Nodes:** Explicitly represent interaction terms between multiple protected attributes to capture unique patterns

```mermaid
graph LR
    %% Protected Attribute
    A[A]
    
    %% Mediators/Compounders
    M1[M1]
    M2[M2]
    M3[M3]
    M4[M4]
    M5[M5]
    
    %% Proxies
    P1[P1]
    P2[P2]
    
    %% Intermediate Outcome
    Y1[Y1]
    
    %% Final Outcome
    Y2([Y2])
    
    %% Causal pathways (solid arrows)
    A -->|causal_1| M1
    A -->|causal_2| M2
    M1 -->|causal_3| Y1
    M2 -->|causal_4| M3
    M3 -->|causal_5| Y1
    M4 -->|causal_6| Y1
    M5 -->|causal_7| Y1
    Y1 -->|causal_8| Y2
    P1 -->|causal_9| M4
    P2 -->|causal_10| M5
    
    %% Correlations (dashed arrows)
    A <-.->|correlation_1| P1
    A <-.->|correlation_2| P2
    
    
    %% Styling
    classDef protected fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    classDef mediator fill:#4dabf7,stroke:#1971c2,stroke-width:2px
    classDef proxy fill:#ffd43b,stroke:#fab005,stroke-width:2px
    classDef outcome fill:#69db7c,stroke:#2f9e44,stroke-width:3px
    
    class A protected
    class M1,M2,M3,M4,M5 mediator
    class P1,P2 proxy
    class Y1,Y2 outcome
   ``` 

**Documentation Template:**

| Arrow    | Causal Assumptions                                | 
| -------- | ------------------------------------------------- | 
| causal_1 | [Document causal assumptions with justifications] |
| causal_2 | [Document causal assumptions with justifications] |
| causal_3 | [Document causal assumptions with justifications] |

**Discrimination Assessment**
   Distinguish between different types of discrimination that may appear statistically identical
   - **Direct Discrimination:** 
     * [ ]  Is there a path “Protected Attribute → Outcome”?
   - **Indirect Discrimination:** 
     * [ ]  Is there a path through mediators (e.g., Protected Attribute → Education → Outcome)?
     * [ ]  Are these mediators considered "legitimate" predictive relationships (like debt-to-income ratio) or do they merely reflect historical inequities (like biased standardized test scores)?
   - **Proxy Discrimination:** 
     * [ ]  Are there variables correlated with the attribute through shared ancestors (e.g., Zip Code)?
     * [ ]  Do these associations exist due to unmeasured common causes, such as historical residential segregation or societal expectations?
     * [ ]  If you controlled for all individual financial factors, would the proxy variable still have a direct causal relationship with the outcome? If not, it is likely a discriminatory proxy.

## 3 Formalization with Structural Equation Models (SEMs)

<details>
   <summary>
   Guidelines
   </summary>
   Kusner et al. (2017) employ SEMs as the foundation for counterfactual fairness in their landmark paper. They demonstrate how these models enable precise definition of counterfactual scenarios by showing how interventions on protected attributes would propagate through the system according to the specified causal structure.
   Provide the mathematical formalization of the relationships mapped in the DAG to transit from qualitative causal mapping (graphs) to the quantitative modeling required for Counterfactual Fairness.

   | Component                | Description                                                                                                                | Specification Format |
   | ------------------------ | -------------------------------------------------------------------------------------------------------------------------- | -------------------- |
   | Exogenous Variables (U)  | Unobserved background factors or error terms $U = \{U_1, U_2, \ldots, U_n\}$ where $P(U)$ is the probability distribution.           |                      |
   | Endogenous Variables (V) | Observed variables (protected attributes A, features X, and outcomes Y) $V = \{A, X, Y\}$.                                     |                      |
   | Structural Equations (F) | Mathematical functions determining the value of each endogenous variable $V_i = f_i(\text{Parents}(V_i), U_i)$                          |                      |
   | Causal Graph (G)         | The Directed Acyclic Graph (DAG) representing these dependencies $G = (V, E)$ where edges $E$ represent direct causal effects. |                      |
</details>

For every endogenous variable (V) - node in your causal graph, specify the following:

| Variable ($V_i$)      | Direct Causes (Parents) | Exogenous Factor ($U_i$) | Structural Equation $f_i($Parents$, U_i)$ |
| --------------------- | ----------------------- | ------------------------------ | ----------------------------------- |
| Example: Credit Score | Race, Income, Education | Unmeasured financial behaviors | $V_i = f_i(\text{Parents}(V_i), U_i)$ |
| Outcome ($Y$)         | [List Parents]          | [Unobserved Noise]             | $Y = f(A, M, C, U_y)$                  |

## 4 Validate the Causal Model

<details>
   <summary>
   Guidelines
   </summary>
   Validation acts as a **stress-test** for the "blueprint" (the graph) and the "recipe" (the equations) before they are used to guide interventions.

   **Validation Protocols**
   1. **Conditional Independence Testing (d-separation):** Test whether the data reflects the statistical independencies implied by the graph. For example, if the model implies that Zip Code and Education are independent given Income, verify this statistically; significant deviations suggest the structure needs revision.
   2. **Sensitivity Analysis (E-Value):** Use metrics like the **E-value** to quantify how strong an unmeasured confounder would need to be to nullify the observed causal effect. If such a strong confounder is implausible in the real world, the model is considered robust.
   3. **Expert and Literature Review:** Compare model predictions and structures with established domain findings and consult with subject matter experts to evaluate the plausibility of the hypothesized pathways.
   4. **Comparative Evaluation:** Apply multiple inference methods (e.g., matching vs. instrumental variables) to the same data. If different methods yield divergent results, it identifies conditions where conclusions are sensitive to methodological choices.

</details>

**Documentation Template: Validation Evidence and Refinement**

| Validation Step       | Test Performed      | Results/Evidence                               | Refinement Action Taken                              |
| --------------------- | ------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| Independence Test     | $A \perp Y \mid M$  | P-value < 0.05 (Independence failed)           | Added missing edge from $A$ to $Y$ (Direct Bias).    |
| Sensitivity           | E-value calculation | E-value = 2.5                                  | Documented as "Robust to moderate confounding."      |
| Expert Review         | Stakeholder Panel   | Expert noted missing "Neighborhood" node.      | Iteratively refined DAG to include Confounder $C$.   |
| Predictive Validation | Prospective testing | Intervention matched 85% of predicted results. | Confirmed model's ability to predict policy changes. |


**Continuous Refinement Protocol**

Causal models should be treated as **iterative**. The framework must include a scheduled review where new observational data or natural experiments (e.g., policy changes) are used to update the structural equations and re-test independencies.

**Documentation: Model Assumptions and Limitations**

A formal record must be kept to ensure transparency for stakeholders:

- **Causal Assumptions:** List every assumed link and categorize its source (e.g., "Domain Expertise," "Historical Literature," or "Data-Driven Discovery").
- **Uncertainty Inventory:** 
  - Explicitly document potential **unobserved confounders**—variables not in the dataset that might affect both the attribute and outcome.
  - Limited Domain Knowledge: Areas where the causal structure is hypothetical.
- **Confidence Levels:** Assign a confidence score to each structural relationship to indicate the degree of certainty in the hypothesized mechanism.


## 5 Counterfactual Query Framework

<details>
   <summary>
   Guidelines
   </summary>
   Per Kusner et al. (2017), a prediction Ŷ is counterfactually fair if the distribution of the prediction is the same in the actual world as it would be in a counterfactual world where the individual’s protected attribute was changed.
   Mathematical Definition:
   $$P(\hat{Y}_{\{A \leftarrow a\}}(U) = y | X = x, A = a) = P(\hat{Y}_{\{A \leftarrow a'\}}(U) = y | X = x, A = a)$$     
</details>

**Counterfactual Formulation Template:**

| Component                | Description & Guidance                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The Factual Case         | Document the actual observed features (X), the original protected attribute (A=a), and the model’s prediction (Y)                                      |
| The Counterfactual Query | Formulate the question: "Would [Applicant Name] have been [Approved/Denied] if their [Gender/Race] had been [Counterfactual Value]?"                   |
| Abduction Step           | Update the unobserved background variables (U)—such as underlying financial responsibility—based on the observed evidence of this specific individual. |
| Action Step              | Theoretically intervene by changing the protected attribute to the counterfactual value (e.g., Female → Male) in your causal model.                    |
| Prediction Step          | Compute the new distribution for the target variable (the loan decision) using the updated attribute and background variables.                         |
| Fairness Implication     | Compare the factual and counterfactual results. A difference in outcome indicates a counterfactual fairness violation.                                 |

## 6 Pathway Classification Template: 

This framework categorizes the causal paths through which a protected attribute influences an outcome. This recognizes that not all influences of protected attributes constitute discrimination; some may represent legitimate predictive relationships
based on ethical principles, domain knowledge, and stakeholder input

| Pathway Category | Description                                                                     | Examples (Lending Domain)                 | Rationale / Documentation Requirement                                                     |
| ---------------- | ------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| Unfair           | Direct paths from attribute to outcome or paths through non-legitimate proxies. | Race → Zip Code → Loan Decision.          | No plausible relation to repayment ability; reflects historical segregation.              |
| Fair             | Paths through mediators that represent legitimate predictors of success/risk.   | Income → Debt-to-Income Ratio → Approval. | Variables directly related to the applicant's ability to repay the loan.                  |
| Contested        | Pathways where stakeholders disagree on fairness vs. utility.                   | Education → Income → Credit History.      | Requires normative judgment: Does it reflect structural bias or a valid skill difference? |

Stakeholder Rationale Documentation: [Teams must record the justification for each classification based on domain expertise, ethical reasoning, and regulatory requirements.Disagreements between stakeholders should be documented to highlight varying assumptions about discrimination.]


## 7 Counterfactual Evaluation Metrics: 

To quantify the degree of fairness violations, the toolkit utilizes metrics that measure the disparity between factual and counterfactual worlds.
1. Counterfactual Effect Size:
    ◦ Definition: Measures the average difference in predictions between the factual and counterfactual scenarios across the dataset.
    ◦ Usage: Used to quantify the total magnitude of bias present in the model's decision-making process.
2. Counterfactual Fairness Violation Rate:
    ◦ Definition: The proportion of instances where the counterfactual prediction differs from the factual one by more than a specified threshold.
    ◦ Usage: Indicates how widespread unfairness is across the population being evaluated.
3. Path-Specific Effect Measures:
    ◦ Definition: Metrics that decompose the total counterfactual effect to isolate the influence along specific causal pathways classified as unfair.
    ◦ Usage: Used to verify that interventions have successfully blocked problematic pathways (e.g., zip code) while preserving legitimate ones (e.g., income).
4. Counterfactual Disparity Measures (Intersectional):
    ◦ Definition: Assessing effect sizes and violation rates across overlapping demographic intersections (e.g., Black women) rather than single attributes.
    ◦ Usage: Ensures that fairness is achieved for subgroups that may experience unique patterns of disadvantage not visible in aggregate data.

## 8 Method Selection Framework
   This framework provides decision criteria for choosing the most appropriate observational causal inference method based on your data’s characteristics and the depth of your domain knowledge.

   | Method                          | Best Applied When...                                                                                                                   | Key Assumptions                                                                                                                |
   | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
   | Matching / Propensity Score     | Rich covariate (feature) data is available to compare individuals with similar characteristics but different protected attributes.     | No Unmeasured Confounding: All variables affecting both the protected attribute and the outcome are measured.                  |
   | Instrumental Variables (IV)     | You can identify an external "instrument" that influences the protected attribute but affects the outcome only through that attribute. | Exclusion Restriction: The instrument has no direct effect on the outcome and is not correlated with unobserved confounders.   |
   | Regression Discontinuity (RD)   | Threshold-based policies (e.g., a specific credit score cutoff) create natural experiments at the boundary.                            | Local Randomization: Individuals just above and below the threshold are otherwise identical in their characteristics.          |
   | Difference-in-Differences (DiD) | Longitudinal data is available that spans a specific policy change or intervention across different groups.                            | Parallel Trends: In the absence of the intervention, the difference between the groups would have remained constant over time. |

   Decision Guidance: Whenever possible, you should combine multiple complementary methods to assess the consistency of your conclusions across different sets of assumptions

## 9 Sensitivity Analysis Approach

   Sensitivity analysis quantifies the robustness of your causal conclusions against potential unmeasured confounding—variables like "generational wealth" that might not be in your dataset but influence both race and loan default risk.
   • The E-value Procedure:
      1. Calculate the Observed Effect: Measure the disparity in your model (e.g., the risk ratio of denial between groups).
      2. Compute the E-value: Determine the minimum strength that an unmeasured confounder would need to have with both the protected attribute and the outcome to explain away the observed effect.
      3. Domain Validation: Consult with subject matter experts to determine if a confounder of that magnitude is plausible.
   • Interpretation Template:
      ◦ "To nullify the observed disparity in loan approvals, an unmeasured factor (e.g., quality of legal representation) would need to be associated with both gender and approval with a risk ratio of at least X.X".
      ◦ If X.X is high (e.g., 2.5+), the conclusion of discrimination is considered robust. If it is low (e.g., 1.1), the conclusion is sensitive and requires more data.

## 10 Uncertainty Communication Framework

   Stakeholders often expect definitive answers, but causal inference from observational data is inherently limited. This framework ensures transparency and manages expectations.
   Guidelines for Communication:
   • Present Bounded Estimates: Instead of point estimates (e.g., "The bias is 12%"), provide ranges or bounds that reflect different plausible scenarios of unmeasured confounding.
   • Risk Management Framing: Frame fairness interventions as risk management strategies rather than proofs of guilt, especially when causal certainty is limited.
   • Visualization Approaches:
      ◦ Use conveyance of uncertainty tools, such as plots showing how the estimated bias shifts as the assumed strength of unmeasured confounders increases.
      ◦ Highlight intersectional uncertainty, showing where data is sparse for specific subgroups (e.g., older minority women) and where conclusions are therefore less certain.
   Stakeholder Education: Explicitly educate technical and non-technical stakeholders that the "absence of conclusive evidence" for discrimination is not "evidence of its absence," particularly when data limitations constrain the analysis.

## 11 Intervention Point Identification Method

   The Intervention Point Identification Method helps determine where in the ML pipeline to intervene based on causal analysis results. This method maps causal patterns to appropriate intervention types through a decision tree.

   **Intervention Selection Decision Tree**

   ```mermaid
   graph TD
      A["Direct Discrimination Detected?"] -->|Yes| B["Protected Attribute<br/>Explicitly Used?"]
      A -->|No| C["Proxy Discrimination<br/>Detected?"]
      
      B -->|Yes| D["Remove Attribute /<br/>In-processing Constraints"]
      B -->|No| E["Investigate Implicit<br/>Discrimination in<br/>Model Architecture"]
      
      C -->|Yes| F["Proxy Variables<br/>Identifiable?"]
      C -->|No| G["Mediator Discrimination<br/>Detected?"]
      
      F -->|Yes| H["Pre-processing:<br/>Transform Proxy<br/>Variables"]
      F -->|No| I["In-processing:<br/>Regularization to<br/>Minimize Proxy Use"]
      
      G -->|Yes| J["Mediators are<br/>Legitimate Predictors?"]
      G -->|No| K["Outcome Discrimination<br/>Detected?"]
      
      J -->|Yes| L["Multi-objective Optimization:<br/>Balance Fairness &<br/>Prediction"]
      J -->|No| M["Pre-processing:<br/>Remove Protected<br/>Attribute Influence"]
      
      K -->|Yes| N["Disparities Consistent<br/>Across Subgroups?"]
      K -->|No| O["End: No Discrimination<br/>Detected"]
      
      N -->|Yes| P["Post-processing:<br/>Threshold Optimization"]
      N -->|No| Q["Targeted Interventions<br/>by Causal Structure"]   
   ```
