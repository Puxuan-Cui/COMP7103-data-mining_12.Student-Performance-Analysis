# COMP7103-data-mining_12.Student-Performance-Analysis
This the assignment1 for COMP7103 data mining.
All code can be run directly after installing the corresponding library, without the need for further debugging.
The file "Student.csv" is the dataset originally selected from the Kaggle website.
In data pre-processing, use Pandas' dataframe function to display various attributes of the dataset and check its integrity.
The data set obtained after data preprocessing is named "exams.csv", and the subsequent data visualization and machine learning model implementation are based on this new data set.

In model implemenatation, After randomly dividing the training and testing sets of the dataset, three functions of the sklearn library were called to construct the linear regression, decision tree regression, and random forest regression models.
After the model training is completed, the model will be evaluated by outputting mse, mae, r-square.
The project also produced residual plots to better visualise the difference between predicted and actual values.
And a feature imporatance plot to analyze the model.
