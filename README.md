# WaterQuality ML Project

**Machine Learning models for predicting and analyzing global surface water quality (1940–2023)**  
This project uses the open-access dataset by *Karim et al. (2025)* to predict the **Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI)** and explore further pollutant interrelationships.

---

## Setup Instructions

Follow the steps to run the project locally:

1. **Get the dataset**

Since the dataset is larger than 25 MB, it can be found and downloaded through the following link:
https://figshare.com/articles/dataset/A_Comprehensive_Surface_Water_Quality_Monitoring_Dataset_1940-2023_2_82Million_Record_Resource_for_Empirical_and_ML-Based_Research/27800394?utm_source=chatgpt.com&file=50757321
   
3. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/WaterQuality.git
   cd WaterQuality

4. **Create and activate a virtual environment**

   python -m venv venv
   
   *Activate the environment*
  
   source venv/bin/activate      # For Mac/Linux

   venv\Scripts\activate         # For Windows

5. **Install the dependencies**
Python required, then run:
pip install -r requirements.txt

6. **Download the dataset**
The dataset is publicly available at:
https://doi.org/10.6084/m9.figshare.27800394.v2

7. **Run the baseline model**
Jupyter notebook:
notebooks/01_baseline_WQI_prediction.ipynb






