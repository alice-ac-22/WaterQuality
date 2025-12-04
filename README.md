This project focuses on predicting and analyzing surface water quality using machine learning techniques.
It utilizes publicly available dataset containing long-term records of water quality parameters, merged, cleaned, and analyzed to assess trends and build predictive models for water quality (CCME values and Water Quality Index). The project includes data cleaning, preprocessing, feature engineering, regression models, classification models, clustering (PCA + K-means), and an experimental LSTM model.

**Dataset**

Since the dataset is larger than 25 MB, it can be downloaded from:

A Comprehensive Surface Water Quality Monitoring Dataset (1940–2023) – Figshare (https://figshare.com/articles/dataset/A_Comprehensive_Surface_Water_Quality_Monitoring_Dataset_1940-2023_2_82Million_Record_Resource_for_Empirical_and_ML-Based_Research/27800394)

After downloading, place the CSV file (e.g., Combined_dataset.csv) in the root project directory (*main project folder*).



**Setup Instructions**

*1. Clone the repository*

        git clone https://github.com/alice-ac-22/WaterQuality.git
        
        cd WaterQuality


*2. Install dependencies*

        This project uses common Python libraries for data analysis and machine learning.
        If any import errors appear, install the missing libraries, such as:

        pip install scikit-learn
        
        pip install seaborn
        
        pip install tensorflow (only needed for the LSTM model)

*3. Get the dataset*

        Download the dataset using the link above and place it in the *main project folder* (where the file Combined_dataset.csv is stored).

*4. Run the project*

        You can run the main workflow from the notebook:

        MAIN.ipynb
        
        Store the needed python .py files (functions) available in the main branch of this repository in the *main project folder*, so that they are directly called upon             when running MAIN.ipynb.


**Repository Structure**

├── MAIN.ipynb                             # Main notebook to run and orchestrate the workflow

├── cleaning_rawdata.py                    # Data cleaning and preprocessing steps

├── preprocessing_utils.py                 # Encoding, scaling, splitting utilities

├── data_visualization.py                  # Exploratory data analysis & plotting functions

│

├── run_model_1.py                         # Regression model to predict CCME Values

├── run_model_2_softmax_regression.py      # Multiclass classification model

├── model_2_tuning.py                      # Hyperparameter tuning for Model 2

├── run_model_3_parameter_prediction.py    # Regression model predicting water parameters

├── model_3_final.py                       # Final version of Model 3

├── run_model_4.py                         # PCA + K-means clustering (unsupervised)

├── run_model_LSTM.py                      # LSTM sequential model (experimental)

│

└── README.md

