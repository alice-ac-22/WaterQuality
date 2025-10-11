# WaterQuality ML Project

**Machine Learning models for predicting and analyzing global surface water quality (1940–2023)**  
This project uses the open-access dataset by *Karim et al. (2025)* to predict the **Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI)** and explore further pollutant interrelationships.

---

## 📁 Project Structure

| Folder | Purpose |
|--------|----------|
| `data/` | Contains sample data or references to dataset location. |
| `notebooks/` | Step-by-step Jupyter notebooks for each phase of the project. |
| `src/` | Modular Python code for preprocessing, modeling, and visualization. |
| `results/` | Stores output plots, metrics, and tables for reporting. |
| `requirements.txt` | Lists Python dependencies for easy environment setup. |
| `README.md` | Overview, setup guide, and project explanation. |

---

**P.S.** Keep dataset files (.csv) in `.gitignore` if they’re too large, and link to the public source instead.


---

## Setup Instructions

Follow the steps to run the project locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/WaterQuality.git
   cd WaterQuality

2. **Create and activate a virtual environment**

   python -m venv venv
   
   *Activate the environment*
  
   source venv/bin/activate      # For Mac/Linux

   venv\Scripts\activate         # For Windows

4. **Install the dependencies**
Python required, then run:
pip install -r requirements.txt

5. **Download the dataset**
The dataset is publicly available at:
https://doi.org/10.6084/m9.figshare.27800394.v2

6. **Run the baseline model**
Jupyter notebook:
notebooks/01_baseline_WQI_prediction.ipynb






