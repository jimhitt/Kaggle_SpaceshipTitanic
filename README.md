# Kaggle Spaceship Titanic Competition

This is a GitHub repository for various Jupyter Notebooks used with the Kaggle Titanic Spaceship dataset.



## Getting Started

Not sure yet. I will include a requirements.txt file.
It basically needs the typical data science items (numpy, pandas, matplotlib, etc ...)
tensorflow and XGBoost for ML

### Notebooks
#### spaceship-titanic-exaple.ipynb 
- This was taken from Kaggle.
#### data_exploration.ipynb
- Includes the initial exploratory data analysis
- Uses XGBoost to make predictions
- Uses a Deep Neural Network (without performing any data scaling)
#### explore_RobustScaler.ipynb
- This examined the numeric data columns.
- There are a large number of zero values with significant rightward skew in the data.
- RobostScaler seems to partially take care of this.
- Explores the binning of the numeric data (besides age) to create categorical data
#### DeepNeuralNet_v*.ipynb
- Develops a Deep Neural Network that initially underperforms
- Experiments with Dropping and L2-regularization
- Experiments with different combinations of the numeric spending columns
- Ultimately, the DNN in v4 performs the best
#### H20_test*.ipynb
- Explores the use of H2O AutoML.
- This creates complicated ensemble models that ultimately do not improve on the DNN

### Prerequisites

These notebooks were created in Visual Studio Code using Jupyter notebooks.

The dependncies are listed in the ```dependencies.txt``` file. Some notebooks use H2O AutoML, which also requires Java.

## Data are from Kaggle

* [Spaceship Titanc](https://www.kaggle.com/competitions/spaceship-titanic) - The competition website
Kaggle is a great site for data sets and competitions. The official competitions are over my head, but this is a great starting project


## Authors

* **James Hitt** - *Initial work* - [LinkedIN](https://www.linkedin.com/in/jim-hitt-mdphd/)


## License

This project is licensed under the MIT License 

## Acknowledgments

* Kaggle is a great site for data science exploration
* **ChatGPT** and **Copilot** have been very helpful taking my analysis ideas and making it happen with code. These tools cannot complete the project on their own, but they are very helpful for completing code segments. This was my first time using assistance with writing code, and it saved tremendous amounts of time.
