# Math533_Hw2
CSUF Math 533 HW 2 

This assignment will guide you through regression modeling and model validation using the dataset alzheimer_data.csv. You will apply multiple regression, logistic regression, and multinomial logistic regression, along with cross-validation techniques for model selection and evaluation.

1. Multiple Regression Analysis
Task: Build and evaluate a multiple regression model where the response variable is the ratio of left hippocampus volume to total intracranial volume.

Begin with a saturated model that includes the following predictors:

Cognitive and behavioral measures: NACCMMSE, MOTSEV, DISNSEV, ANXSEV, NACCGDS

Physiological measures: BPSYS, BPDIAS, HRATE

Demographics: Age, Educ, Female, Height, Weight

Perform model selection using:

Leave-one-out cross-validation (LOOCV)

Cross-validation with a single fold (training/test split)

Cross-validation with 5 folds

Cross-validation with 10 folds

Compare the predictive performance of the different models.

Interpret the results of your final selected model, discussing both prediction accuracy and the meaning of the estimated coefficients.

2. Logistic Regression (Binary Outcome)
Task: Build and evaluate a logistic regression model where the response variable is a binarized version of diagnosis. Specifically, recode the original variable so that levels 1 and 2 are coded as 1, and all other levels are coded as 0.

Fit an initial logistic regression model using relevant predictors.

Perform 5-fold cross-validation for model validation and selection.

Interpret the coefficients of your final model, explaining how each predictor influences the probability of Alzheimer’s diagnosis.

3. Multinomial Logistic Regression (Three Categories)
Task: Extend your analysis by modeling the original diagnosis variable, which has three categories, using multinomial logistic regression.

Fit a multinomial logistic regression model and carry out model selection.

Use appropriate model validation techniques to evaluate predictive performance.

Interpret the coefficients of your final multinomial model, and compare these results with your findings from Part 2. Highlight similarities and differences in how predictors influence diagnosis under the binary versus multinomial frameworks.

Deliverables
Your submission must be a Quarto report (.qmd) that can be rendered into either HTML or PDF. The .qmd file must be fully reproducible and run on any computer (i.e., include all necessary code, packages, and clear instructions).

Your report should include the following:

Modeling Process and Decisions

A clear explanation of your modeling workflow, including data preparation, model specification, and selection criteria.

Comparative Results Across Validation Strategies

Summarize and compare model performance under different cross-validation approaches (LOOCV, single fold, 5-fold, 10-fold).

Provide appropriate tables and/or figures to support your comparison.

Interpretation of Model Coefficients

Explain the meaning of the coefficients from the final selected models, connecting results to the context of Alzheimer’s research.

Discussion of Modeling Approaches

Reflect on the strengths and limitations of linear regression, binary logistic regression, and multinomial logistic regression.

Discuss how these approaches provide complementary insights into the data.

This is a Group Assignment
This assignment is a group project. Groups were randomly generated in Canvas. To see your assigned group, go to the People tab on the left navigation bar in Canvas. Within People, you will find a group set called Homework Buddies.

Each group has a designated leader, who is responsible for:

Uploading the group’s completed homework to Canvas.

Communicating with me directly in case any issues arise.

Please coordinate with your group members early to ensure smooth collaboration.