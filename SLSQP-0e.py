#!/usr/bin/env python
# coding: utf-8

# # SLSQP Optimization

# **To avoid confusion, this notebook requires several different datasets:**
# 
# 1. **`df_to_optimize`**:  
#    This is the master dataset, containing all the cleaned and preprocessed data. It is used to calculate distribution-related values such as the Interquartile Range (IQR) and Standard Deviation.
# 
# 2. **`df_filtered`**:  
#    This dataset is a subset of the master data that satisfies the defined constraints. Not all entries in the full dataset meet these constraints. `df_filtered` is used to create a scaler that can properly invert `df_test`, since `df_test` is received in a normalized form.
# 
# 3. **`df_test`**:  
#    This dataset is a subset of `df_filtered`, specifically 20% of it. During the model training process, the model is trained using `df_filtered` — a "perfect" dataset that fully aligns with the constraints. `df_test` is the dataset that will be optimized in this notebook. This dataset is a merge between `df_x_test_inconstraint` and `df_y_test_inconstraint`

# ## Library

# This section provides instructions to install the Python libraries required for the **FeNi Production System** code, which handles data processing, machine learning, and optimization tasks. The following libraries are used:
# 
# - **pandas**: For data manipulation and analysis.
# - **numpy**: For numerical computations and array operations.
# - **scikit-learn**: For machine learning models, preprocessing, and evaluation metrics (includes `MinMaxScaler`, `Ridge`, `train_test_split`, `mean_squared_error`, `mean_absolute_error`, `r2_score`, `mean_absolute_percentage_error`).
# - **scipy**: For optimization tasks (includes `minimize`).
# - **joblib**: For loading saved models or scalers.
# 
# ### Installation Steps
# 
# 1. **Ensure Python is Installed**
# 
#    Make sure you have Python (version 3.6 or higher) installed. You can check this by running:
# 
#    ```bash
#    python --version
#    ```
# 
# 2. **Install the Required Libraries**
# 
#    Use `pip` to install the libraries. Run the following command in your terminal or command prompt:
# 
#    ```bash
#    pip install pandas numpy scikit-learn scipy joblib
#    ```

# In[77]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.optimize import minimize
import os
from joblib import load
pd.set_option('display.max_columns', None)  # Show all columns


# In this section, we import the filtered data as `df_filtered` and scale each column individually. A separate scaler is stored for each column in an array. These scalers will later be used to invert a dataset, which will be explained in a following section.

# In[78]:


df_filtered = pd.read_csv('../dataframe/df_filtered.csv')
df_filtered.describe()


# In[79]:


scalers = {}
scaled_data = {}

for col in df_filtered:
    scaler = MinMaxScaler()
    scaled_data[col] = scaler.fit_transform(df_filtered[[col]])[:, 0]  # Scale and flatten
    scalers[col] = scaler  # Store the scaler for later inverse transform


# These test files, which are also imported, are split portions of the full filtered dataset that will be optimized. They were saved as separate files as a result of the `train_test_split` function during the training process. These files are then merged into a single dataset called `df_test` for easier processing during optimization.
# 
# `df_test` is already normalized; therefore, it must be inverted to retrieve the actual values of the data.

# In[80]:


# Load the CSV files
df_x_test_inconstraint = pd.read_csv("../dataframe/df_x_test_inconstraint.csv", encoding="unicode_escape")
df_y_test_inconstraint = pd.read_csv("../dataframe/df_y_test_inconstraint.csv", encoding="unicode_escape")

df_x_test_inconstraint = df_x_test_inconstraint.drop(columns=["row_number"])
df_y_test_inconstraint = df_y_test_inconstraint.drop(columns=["row_number"])

# Merge the dataframes horizontally
df_test_normalized = pd.concat([df_x_test_inconstraint, df_y_test_inconstraint], axis=1)

# Display the merged dataframe
df_test_normalized


# In[81]:


df_test = pd.DataFrame({col: scalers[col].inverse_transform(df_test_normalized[[col]])[:, 0] for col in df_test_normalized.columns})
df_test


# In[82]:


df_test.to_csv("../dataframe/df_test.csv", index=False)


# In this section, we import the `df_to_optimize` dataset. This dataset contains all available data that has been cleaned and preprocessed—essentially serving as the master dataset.
# 
# We also calculate the Interquartile Range (IQR) and Standard Deviation for each column and store the values in separate arrays for easy access.

# In[83]:


df_to_optimize = pd.read_csv("../dataframe/df_to_optimize.csv", encoding="unicode_escape")


# In[84]:


df_to_optimize.describe()


# In[85]:


# Calculate the IQR for each column
iqr_df = df_to_optimize.quantile(0.75) - df_to_optimize.quantile(0.25)

# Convert the result to a DataFrame for better readability
iqr_df = iqr_df.to_frame(name='IQR').reset_index()
iqr_df.rename(columns={'index': 'Column'}, inplace=True)

# Display the IQR DataFrame
print(iqr_df)


# In[86]:


# Calculate the standard deviation for each column
std_df = df_to_optimize.std()

# Convert the result to a DataFrame for better readability
std_df = std_df.to_frame(name='Standard Deviation').reset_index()
std_df.rename(columns={'index': 'Column'}, inplace=True)

# Display the standard deviation DataFrame
print(std_df)


# ## Model Permormance Check

# In this section, we import the `Ridge Model`. We also normalize and split the `df_test` to check the performance of the `Ridge Model` once again

# In[87]:


input_cols = df_to_optimize.loc[:, 'ni_in':'t_tic163'].columns
output_cols = df_to_optimize.loc[:, 'metal_temp':'loi_kalsin'].columns


# In[88]:


# Initialize Ridge Regression model
ridge_model = load("../model/ridge_model.pkl")

# X_merged = pd.concat([normalized_df[input_cols], pca_df], axis=1)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_norm = pd.DataFrame(scaler_x.fit_transform(df_test[input_cols]), columns=input_cols)
y_norm = pd.DataFrame(scaler_y.fit_transform(df_test[output_cols]), columns=output_cols)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
X_train


# In[89]:


X_test.columns


# In[90]:


# Check if the model has the attribute `feature_names_in_`
if hasattr(ridge_model, 'feature_names_in_'):
    print("The Ridge model was trained on the following columns:")
    print(ridge_model.feature_names_in_)
else:
    print("The Ridge model does not have the `feature_names_in_` attribute. It might not store the column names.")


# In[91]:


# Predict on test set
y_pred_rr = ridge_model.predict(X_test.to_numpy())
y_pred_rr = scaler_y.inverse_transform(y_pred_rr)
y_pred_rr_df = pd.DataFrame(y_pred_rr, columns=output_cols)
X_test_inverted = scaler_x.inverse_transform(X_test)
X_test_inverted = pd.DataFrame(X_test_inverted, columns=input_cols)
inverted_pred_rr_df = pd.concat([X_test_inverted.reset_index(drop=True), y_pred_rr_df.reset_index(drop=True)], axis=1)
inverted_pred_rr_df


# In[92]:


inverted_y_actual = scaler_y.inverse_transform(y_test)
inverted_y_actual = pd.DataFrame(inverted_y_actual, columns=output_cols)
inverted_y_pred_rr = y_pred_rr_df[output_cols]
inverted_y_pred_rr


# In[93]:


models = {
    'RR': inverted_y_pred_rr,

}
mae_dict = {'Column': output_cols}
# Iterate through each model and calculate MAE for each column
for model_name, inverted_y_pred in models.items():
    mae_dict[model_name] = []  # Add a column for each model
    for col in output_cols:
        # Extract actual and predicted values
        y_actual_col = inverted_y_actual[col]
        y_pred_col = inverted_y_pred[col]

        # Calculate MAE and store in the dictionary
        mae_col = mean_absolute_error(y_actual_col, y_pred_col)
        mae_dict[model_name].append(mae_col)

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(mae_dict)

# Display the DataFrame
display(metrics_df)


# ## Constraint Definition

# ### Output Constraint

# In this section, we define constraints for the output variables of the FeNi (Ferronickel) Production System by setting lower and upper bounds. These variables represent the **system's outputs**, such as metal temperature and chemical compositions, but will later act as input variables during optimization to predict or find the optimal system inputs. The constraints are based on provided information, such as target values or expected ranges. Since many data points we collected fell outside the original constraints, we added tolerances (using the Interquartile Range (IQR) or Standard Deviation (Std)) to make the bounds more flexible and realistic, ensuring they better reflect the actual data.
# 
# 1. **Retrieve Min and Max Values from Scaler**  
#    - `min_values` and `max_values` store the minimum and maximum values used by the output scaler (`scaler_y`).
#    - A list of *bounds* is created by pairing each minimum and maximum value.
# 
# 2. **Set Target or Threshold Values for Each Output Column**  
#    For each output variable, a target, minimum, or maximum constraint value is specified:
#    - **metal_temp**: Fixed at 1300.
#    - **ni_met**: Lowered by 3 times the Standard Deviation (Std) from a baseline of 17.
#    - **c_met**: Range set between 1 (low) and 3 (high).
#    - **si_met**: Lower bound reduced by 1.5 times the Interquartile Range (IQR) from 1; upper bound fixed at 2.
#    - **fe_met**: Lower bound set to 0.
#    - **s_met**:  
#      - Lower bound set to 0.  
#      - Upper bound increased by 1.5 times the IQR from a baseline of 0.4.
#    - **ni_slag**: Lower bound set to 0.
#    - **fe_slag**: Lower bound set to 0.
#    - **t_kalsin**: Lower bound set to 600.
#    - **pic_161**: Lower bound set to -5.38.
#    - **loi_kalsin**: Range set between 0 (low) and 1 (high).
# 
# 3. **Normalization of Constraint Values**  
#    Each constraint value is normalized using the formula:
#    
#    $ \text{normalized\_value} = \frac{\text{target\_value} - \text{min\_bound}}{\text{max\_bound} - \text{min\_bound}} $
#    
#    This normalization ensures that all constraint values are in the same scale as the normalized model inputs and outputs.

# Since the constraints are derived from the existing data distribution, these constraints may not always align with theoretical expectations but are tailored to fit the existing data. If new data is collected, the constraints may need to be adjusted to remain relevant.

# In[ ]:


min_values = scaler_y.data_min_
max_values = scaler_y.data_max_

print(min_values.shape)
print(max_values.shape)
konstrain=[]
bounds=[ (i,j) for i, j in zip(min_values, max_values)]
print(bounds)

metal_temp=1300
metal_temp_norm= (metal_temp-bounds[0][0])/(bounds[0][1]-bounds[1][0])
print("metal_temp_norm",metal_temp_norm)

ni_met = 17 - 3 * std_df.loc[std_df['Column'] == 'ni_met', 'Standard Deviation'].values[0]
ni_met_norm= (ni_met-bounds[1][0])/(bounds[1][1]-bounds[1][0])
print("Ni_norm",ni_met_norm)

c_met_low=1
c_met_low_norm= (c_met_low-bounds[2][0])/(bounds[2][1]-bounds[2][0])
print("c_met_low_norm",c_met_low_norm)

c_met_high=3
c_met_high_norm= (c_met_high-bounds[2][0])/(bounds[2][1]-bounds[2][0])
print("c_met_high_norm",c_met_high_norm)

si_met_low = 1 - 1.5 * iqr_df.loc[iqr_df['Column'] == 'si_met', 'IQR'].values[0]
si_met_low_norm= (si_met_low-bounds[3][0])/(bounds[3][1]-bounds[3][0])
print("si_met_low_norm",si_met_low_norm)

si_met_high=2
si_met_high_norm= (si_met_high-bounds[3][0])/(bounds[3][1]-bounds[3][0])
print("si_met_high_norm",si_met_high_norm)

fe_met_low=0
fe_met_low_norm= (fe_met_low-bounds[4][0])/(bounds[4][1]-bounds[4][0])
print("fe_met_norm",fe_met_low_norm)

s_met_low=0
s_met_low_norm= (s_met_low-bounds[5][0])/(bounds[5][1]-bounds[5][0])
print("s_met_norm",s_met_low_norm)

s_met_high = 0.4 + 1.5 * iqr_df.loc[iqr_df['Column'] == 's_met', 'IQR'].values[0]
s_met_high_norm= (s_met_high-bounds[5][0])/(bounds[5][1]-bounds[5][0])
print("s_met_high_norm",s_met_high_norm)

ni_slag_low=0
ni_slag_low_norm= (ni_slag_low-bounds[6][0])/(bounds[6][1]-bounds[6][0])
print("ni_slag_low_norm",ni_slag_low_norm)

fe_slag_low=0
fe_slag_low_norm= (fe_slag_low-bounds[7][0])/(bounds[7][1]-bounds[7][0])
print("fe_slag_low_norm",fe_slag_low_norm)

t_kalsin_low=600
t_kalsin_low_norm= (t_kalsin_low-bounds[8][0])/(bounds[8][1]-bounds[8][0])
print("t_kalsin_low_norm",t_kalsin_low_norm)

pic_161_low=-5.28
pic_161_low_norm= (pic_161_low-bounds[9][0])/(bounds[9][1]-bounds[9][0])
print("pic_161_low_norm",pic_161_low_norm)

loi_kalsin_low=0
loi_kalsin_low_norm= (loi_kalsin_low-bounds[10][0])/(bounds[10][1]-bounds[10][0])
print("loi_kalsin_low_norm",loi_kalsin_low_norm)

loi_kalsin_high=1
loi_kalsin_high_norm= (loi_kalsin_high-bounds[10][0])/(bounds[10][1]-bounds[10][0])
print("loi_kalsin_high_norm",loi_kalsin_high_norm)


# In this section, we also define constraints for the **input variables of the FeNi (Ferronickel) Production System**, which control aspects like coal consumption, operational settings, and process parameters. These variables are inputs to the system but some of these variables will later become outputs during the optimization phase, where we aim to find their optimal values to enhance system performance. The constraints are based on the existing data distribution and operational knowledge. To ensure compatibility with the model, we normalize each input constraint using the same formula as for the output columns, keeping the optimized input values within the scaled ranges.
# 
# Some constraints are direct (e.g., `ni_in_min = 0`) and others are dynamic (e.g., percentiles from the current data like `rpm_max`, `pry_p_max`).
# 
# - The total_coal and reductor_consume are pushed down to Q2 value to reduce waste and keep it optimal.
# - Some other parameters also pushed down to Q2 value (`rpm`, `pry_p`, `sec_p`, `pry_v`, `sec_v` and `t_tic162`) because those columns have the highest correlation with total_coal
# - Certain operational settings (e.g., `t_tic162_min`, `t_tic163_min`) are based on fixed minimum values from operational knowledge.
# 

# In[ ]:


x0 = X_norm.iloc[0,:].to_numpy().flatten()
print(x0.shape)
min_values_input = scaler_x.data_min_
max_values_input = scaler_x.data_max_

print(min_values_input.shape)
print(max_values_input.shape)
konstrain_input=[ (i,j) for i, j in zip(min_values_input, max_values_input)]
print(konstrain_input)

# Fixed Values
######################################
ni_in_exact_norm = x0[0]
print("ni_in_exact_norm",ni_in_exact_norm)

fe_in_exact_norm = x0[1]
print("fe_in_exact_norm",fe_in_exact_norm)

cao_in_exact_norm = x0[2]
print("cao_in_exact_norm",cao_in_exact_norm)

al2o3_in_exact_norm = x0[3]
print("al2o3_in_exact_norm",al2o3_in_exact_norm)

s_m_exact_norm = x0[4]
print("s_m_exact_norm",s_m_exact_norm)

bc_exact_norm = x0[5]
print("bc_exact_norm",bc_exact_norm)

mc_kilnfeed_exact_norm = x0[6]
print("mc_kilnfeed_exact_norm",mc_kilnfeed_exact_norm)

fc_coal_exact_norm = x0[7]
print("fc_coal_exact_norm",fc_coal_exact_norm)

gcv_coal_exact_norm = x0[8]
print("gcv_coal_exact_norm",gcv_coal_exact_norm)

fc_lcv_exact_norm = x0[9]
print("fc_lcv_exact_norm",fc_lcv_exact_norm)

gcv_lcv_exact_norm = x0[10]
print("gcv_lcv_exact_norm",gcv_lcv_exact_norm)
#########################################

current_pry_min=0
current_pry_min_norm= (current_pry_min-konstrain_input[11][0])/(konstrain_input[11][1]-konstrain_input[11][0])
print("current_pry_min_norm",current_pry_min_norm)

current_sec1_min=0
current_sec1_min_norm= (current_sec1_min-konstrain_input[12][0])/(konstrain_input[12][1]-konstrain_input[12][0])
print("current_sec1_min_norm",current_sec1_min_norm)

current_sec2_min=0
current_sec2_min_norm= (current_sec2_min-konstrain_input[13][0])/(konstrain_input[13][1]-konstrain_input[13][0])
print("current_sec2_min_norm",current_sec2_min_norm)

current_sec3_min=0
current_sec3_min_norm= (current_sec3_min-konstrain_input[14][0])/(konstrain_input[14][1]-konstrain_input[14][0])
print("current_sec3_min_norm",current_sec3_min_norm)

load_min=0
load_min_norm= (load_min-konstrain_input[15][0])/(konstrain_input[15][1]-konstrain_input[15][0])
print("load_min_norm",load_min_norm)

realisasi_beban_min=0
realisasi_beban_min_norm= (realisasi_beban_min-konstrain_input[16][0])/(konstrain_input[16][1]-konstrain_input[16][0])
print("realisasi_beban_min_norm",realisasi_beban_min_norm)

rpm_min=0
rpm_min_norm= (rpm_min-konstrain_input[17][0])/(konstrain_input[17][1]-konstrain_input[17][0])
print("rpm_min_norm",rpm_min_norm)

rpm_max = np.percentile(df_to_optimize['rpm'], 50)
rpm_max_norm= (rpm_max-konstrain_input[17][0])/(konstrain_input[17][1]-konstrain_input[17][0])
print("rpm_max_norm",rpm_max_norm)

# Fixed Values
############################################
kg_tco_exact_norm = x0[18]
print("kg_tco_exact_norm",kg_tco_exact_norm)

charge_kiln_exact_norm = x0[19]
print("charge_kiln_exact_norm",charge_kiln_exact_norm)

tdo_exact_norm = x0[20]
print("tdo_exact_norm",tdo_exact_norm)
############################################

pry_p_min=0
pry_p_min_norm= (pry_p_min-konstrain_input[21][0])/(konstrain_input[21][1]-konstrain_input[21][0])
print("pry_p_min_norm",pry_p_min_norm)

pry_p_max= np.percentile(df_to_optimize['pry_p'], 50)
pry_p_max_norm= (pry_p_max-konstrain_input[21][0])/(konstrain_input[21][1]-konstrain_input[21][0])
print("pry_p_max_norm",pry_p_max_norm)

sec_p_min=0
sec_p_min_norm= (sec_p_min-konstrain_input[22][0])/(konstrain_input[22][1]-konstrain_input[22][0])
print("sec_p_min_norm",sec_p_min_norm)

sec_p_max= np.percentile(df_to_optimize['sec_p'], 50)
sec_p_max_norm= (sec_p_max-konstrain_input[22][0])/(konstrain_input[22][1]-konstrain_input[22][0])
print("sec_p_max_norm",sec_p_max_norm)

pry_v_min=0
pry_v_min_norm= (pry_v_min-konstrain_input[23][0])/(konstrain_input[23][1]-konstrain_input[23][0])
print("pry_v_min_norm",pry_v_min_norm)

pry_v_max= np.percentile(df_to_optimize['pry_v'], 50)
pry_v_max_norm= (pry_v_max-konstrain_input[23][0])/(konstrain_input[23][1]-konstrain_input[23][0])
print("pry_v_max_norm",pry_v_max_norm)

sec_v_min=0
sec_v_min_norm= (sec_v_min-konstrain_input[24][0])/(konstrain_input[24][1]-konstrain_input[24][0])
print("sec_v_min_norm",sec_v_min_norm)

sec_v_max = np.percentile(df_to_optimize['sec_v'], 50)
sec_v_max_norm= (sec_v_max-konstrain_input[24][0])/(konstrain_input[24][1]-konstrain_input[24][0])
print("sec_v_max_norm",sec_v_max_norm)

total_fuel_min=0
total_fuel_min_norm= (total_fuel_min-konstrain_input[25][0])/(konstrain_input[25][1]-konstrain_input[25][0])
print("total_fuel_min_norm",total_fuel_min_norm)

total_fuel_max=np.percentile(df_to_optimize['total_fuel'], 50)
total_fuel_max_norm= (total_fuel_max-konstrain_input[25][0])/(konstrain_input[25][1]-konstrain_input[25][0])
print("total_fuel_max_norm",total_fuel_max_norm)

reductor_consume_min=0
reductor_consume_min_norm= (reductor_consume_min-konstrain_input[26][0])/(konstrain_input[26][1]-konstrain_input[26][0])
print("reductor_consume_min_norm",reductor_consume_min_norm)

reductor_consume_max=np.percentile(df_to_optimize['reductor_consume'], 50)
reductor_consume_max_norm= (reductor_consume_max-konstrain_input[26][0])/(konstrain_input[26][1]-konstrain_input[26][0])
print("reductor_consume_max_norm",reductor_consume_max_norm)

t_tic162_min=556
t_tic162_min_norm= (t_tic162_min-konstrain_input[27][0])/(konstrain_input[27][1]-konstrain_input[27][0])
print("t_tic162_min_norm",t_tic162_min_norm)

t_tic162_max= np.percentile(df_to_optimize['t_tic162'], 50)
t_tic162_max_norm= (t_tic162_max-konstrain_input[27][0])/(konstrain_input[27][1]-konstrain_input[27][0])
print("t_tic162_max_norm",t_tic162_max_norm)

t_tic163_min=456
t_tic163_min_norm= (t_tic163_min-konstrain_input[28][0])/(konstrain_input[28][1]-konstrain_input[28][0])
print("t_tic163_min_norm",t_tic163_min_norm)

t_tic163_max=845
t_tic163_max_norm= (t_tic163_max-konstrain_input[28][0])/(konstrain_input[28][1]-konstrain_input[28][0])
print("t_tic163_max_norm",t_tic163_max_norm)


# All the constraints that have been defined are stored in an array named *constraints_sample*.
# 
# This code performs optimization on a set of normalized input samples (*X_norm*) to minimize the prediction error of a Ridge Regression model (*ridge_model*) with respect to the true target values (*y_norm*), under a set of specific constraints.

# 1. **Initialization**  
#    - `num_samples` stores the number of samples to optimize (equal to the number of rows in *X_norm*).
#    - `optimized_x_list` and `optimized_y_list` are empty lists prepared to collect the optimized inputs and their corresponding outputs.
# 
# 2. **Iterating Over Each Sample**  
#    A loop goes through each sample in *X_norm*:
#    - `x0` is the normalized input sample.
#    - `x1` is the original (non-normalized) version from *df_test* (but unused in optimization).
#    - `target_value` is the true normalized output value for the sample.
# 
# 3. **Defining the Objective Function**  
#    For each sample, an `objective_function` is defined:
#    - It reshapes the candidate input (*x_scaled*) and predicts the output using the Ridge Regression model.
#    - The objective is to minimize the squared difference between the predicted output and the true target value (essentially minimizing the prediction error).
# 
# 4. **Defining Constraints**  
#    - Certain features must remain exactly the same as their original value (using equality constraints `type: 'eq'`).
#    - Some inputs are required to be greater than or less than certain normalized thresholds (using inequality constraints `type: 'ineq'`).
#    - In addition, **predicted outputs** of the Ridge model must satisfy certain bounds (inequality constraints applied to model predictions).
#    - All these conditions are stored in the list `constraints_sample`.
# 
# 5. **Optimization Process**  
#    - The `minimize` function from `scipy.optimize` is used with the *Sequential Least Squares Programming (SLSQP)* method.
#    - If optimization is successful (`result.success` is True), the optimized input and output are saved.
#    - If optimization fails, `NaN` values are stored for that sample.
# 
# 6. **Post-Processing**  
#    - The optimized inputs and outputs are transformed back to their original scale using the inverse of the scalers (*scaler_x* and *scaler_y*).
#    - These results are saved into pandas DataFrames: *optimized_x_df_original* and *optimized_y_df_original*.
# 
# 7. **Result Display**  
#    - For each sample, if optimization succeeded, the optimized input and output are printed.
#    - If optimization failed, a message is shown indicating the failure.
# 
# 8. **Exporting the Results**  
#    - A unique filename is generated using `generate_unique_filename` to avoid overwriting existing files.
#    - The optimized results, along with the actual inputs and outputs from the test dataset, are exported to an Excel file with two sheets: "Input_Dioptimalkan" and "Output_Dioptimalkan".

# In[ ]:


# Jumlah sampel yang ingin dioptimalkan
num_samples = len(X_norm)

# Inisialisasi list untuk menyimpan hasil optimisasi
optimized_x_list = []
optimized_y_list = []

# Melakukan iterasi melalui semua sampel di X_test
for i in range(num_samples):
    print(f"\nMemproses Sampel ke-{i+1}")
    # Mengambil sampel ke-i dan meratakan array-nya
    x0 = X_norm.iloc[i, :].to_numpy().flatten()
    x1 = df_test.iloc[i, :].to_numpy().flatten()

    # Pastikan y_test hanya memiliki satu kolom output jika prediksi satu fitur
    target_value = y_norm.iloc[i, :].to_numpy()

    # Mendefinisikan fungsi objektif spesifik untuk sampel ke-i
    def objective_function(x_scaled):
        # Mengubah bentuk input untuk prediksi
        x_scaled_reshaped = x_scaled.reshape(1, -1)
        # Melakukan prediksi menggunakan model ridge
        y_pred = ridge_model.predict(x_scaled_reshaped)[0]
        # Menghitung selisih kuadrat (MSE)
        return np.sum(y_pred - target_value) ** 2

    # Mendefinisikan constraints spesifik untuk sampel ke-i
    constraints_sample = [
        # Constraints equality (tidak berubah) (harusnya equality constraint, dikunci berdasarkan data asli)
        {'type': 'eq', 'fun': lambda x, ni_in_exact_norm=ni_in_exact_norm: x[0] - ni_in_exact_norm},  
        {'type': 'eq', 'fun': lambda x, fe_in_exact_norm=fe_in_exact_norm: x[1] - fe_in_exact_norm},
        {'type': 'eq', 'fun': lambda x, cao_in_exact_norm=cao_in_exact_norm: x[2] - cao_in_exact_norm}, 
        {'type': 'eq', 'fun': lambda x, al2o3_in_exact_norm=al2o3_in_exact_norm: x[3] - al2o3_in_exact_norm}, 
        {'type': 'eq', 'fun': lambda x, s_m_exact_norm=s_m_exact_norm: x[4] - s_m_exact_norm}, 
        {'type': 'eq', 'fun': lambda x, bc_exact_norm=bc_exact_norm: x[5] - bc_exact_norm}, 

        # Constraints equality (tidak berubah)
        {'type': 'eq', 'fun': lambda x, mc_kilnfeed_exact_norm=mc_kilnfeed_exact_norm: mc_kilnfeed_exact_norm - x[6]},  # mc_kilnfeed tidak berubah
        {'type': 'eq', 'fun': lambda x, fc_coal_exact_norm=fc_coal_exact_norm: fc_coal_exact_norm - x[7]},  # fc_coal tidak berubah
        {'type': 'eq', 'fun': lambda x, gcv_coal_exact_norm=gcv_coal_exact_norm: gcv_coal_exact_norm - x[8]},  # gcv_coal tidak berubah
        {'type': 'eq', 'fun': lambda x, fc_lcv_exact_norm=fc_lcv_exact_norm: fc_lcv_exact_norm - x[9]},  # fc_lcv tidak berubah
        {'type': 'eq', 'fun': lambda x, gcv_lcv_exact_norm=gcv_lcv_exact_norm: gcv_lcv_exact_norm - x[10]},  # gcv_lcv tidak berubah

        # Constraints input >= min_norm
        {'type': 'ineq', 'fun': lambda x, current_pry_min_norm=current_pry_min_norm: x[11] - current_pry_min_norm},  # current_pry > current_pry_min_norm
        {'type': 'ineq', 'fun': lambda x, current_sec1_min_norm=current_sec1_min_norm: x[12] - current_sec1_min_norm},  # current_sec1 > current_sec1_min_norm
        {'type': 'ineq', 'fun': lambda x, current_sec2_min_norm=current_sec2_min_norm: x[13] - current_sec2_min_norm},  # current_sec2 > current_sec2_min_norm
        {'type': 'ineq', 'fun': lambda x, current_sec3_min_norm=current_sec3_min_norm: x[14] - current_sec3_min_norm},  # current_sec3 > current_sec3_min_norm
        {'type': 'ineq', 'fun': lambda x, load_min_norm=load_min_norm: x[15] - load_min_norm},  # load > load_min_norm
        {'type': 'ineq', 'fun': lambda x, realisasi_beban_min_norm=realisasi_beban_min_norm: x[16] - realisasi_beban_min_norm},  # realisasi_beban > realisasi_beban_min_norm
        {'type': 'ineq', 'fun': lambda x, rpm_min_norm=rpm_min_norm: x[17] - rpm_min_norm},  # rpm > rpm_min_norm
        {'type': 'ineq', 'fun': lambda x, rpm_max_norm=rpm_max_norm: rpm_max_norm - x[17]},  # rpm < rpm_max_norm

        {'type': 'eq', 'fun': lambda x, kg_tco_exact_norm=kg_tco_exact_norm: kg_tco_exact_norm - x[18]},  # kg_tco tidak berubah
        {'type': 'eq', 'fun': lambda x, charge_kiln_exact_norm=charge_kiln_exact_norm: charge_kiln_exact_norm - x[19]},  # charge_kiln tidak berubah
        {'type': 'eq', 'fun': lambda x, tdo_exact_norm=tdo_exact_norm: tdo_exact_norm - x[20]},  # tdo tidak berubah

        {'type': 'ineq', 'fun': lambda x, pry_p_min_norm=pry_p_min_norm: x[21] - pry_p_min_norm},  # pry_p > pry_p_min_norm
        {'type': 'ineq', 'fun': lambda x, pry_p_max_norm=pry_p_max_norm: pry_p_max_norm - x[21]},  # pry_p < pry_p_max_norm
        {'type': 'ineq', 'fun': lambda x, sec_p_min_norm=sec_p_min_norm: x[22] - sec_p_min_norm},  # sec_p > sec_p_min_norm
        {'type': 'ineq', 'fun': lambda x, sec_p_max_norm=sec_p_max_norm: sec_p_max_norm - x[22]},  # sec_p < sec_p_max_norm
        {'type': 'ineq', 'fun': lambda x, pry_v_min_norm=pry_v_min_norm: x[23] - pry_v_min_norm},  # pry_v > pry_v_min_norm
        {'type': 'ineq', 'fun': lambda x, pry_v_max_norm=pry_v_max_norm: pry_v_max_norm - x[23]},  # pry_v < pry_v_max_norm
        {'type': 'ineq', 'fun': lambda x, sec_v_min_norm=sec_v_min_norm: x[24] - sec_v_min_norm},  # sec_v > sec_v_min_norm
        {'type': 'ineq', 'fun': lambda x, sec_v_max_norm=sec_v_max_norm: sec_v_max_norm - x[24]},  # sec_v < sec_v_max_norm
        {'type': 'ineq', 'fun': lambda x, total_fuel_min_norm=total_fuel_min_norm: x[25] - total_fuel_min_norm},  # total_fuel > total_fuel_min_norm
        {'type': 'ineq', 'fun': lambda x, total_fuel_max_norm=total_fuel_max_norm: total_fuel_max_norm - x[25]},  # total_fuel < total_fuel_max_norm
        {'type': 'ineq', 'fun': lambda x, reductor_consume_min_norm=reductor_consume_min_norm: x[26] - reductor_consume_min_norm},  # reductor_consume > reductor_consume_min_norm
        {'type': 'ineq', 'fun': lambda x, reductor_consume_max_norm=reductor_consume_max_norm: reductor_consume_max_norm - x[26]},  # reductor_consume < reductor_consume_max_norm

        # Constraints input >= min_norm dan <= max_norm
        {'type': 'ineq', 'fun': lambda x, t_tic162_min_norm=t_tic162_min_norm: x[27] - t_tic162_min_norm},  # t_tic162 > t_tic162_min_norm
        {'type': 'ineq', 'fun': lambda x, t_tic162_max_norm=t_tic162_max_norm: t_tic162_max_norm - x[27]},  # t_tic162 < t_tic162_max_norm
        {'type': 'ineq', 'fun': lambda x, t_tic163_min_norm=t_tic163_min_norm: x[28] - t_tic163_min_norm},  # t_tic163 > t_tic163_min_norm
        {'type': 'ineq', 'fun': lambda x, t_tic163_max_norm=t_tic163_max_norm: t_tic163_max_norm - x[28]},  # t_tic163 < t_tic163_max_norm

        # Constraints berdasarkan prediksi model Ridge
        {'type': 'ineq', 'fun': lambda x, metal_temp_norm=metal_temp_norm: ridge_model.predict(x.reshape(1, -1))[0][0] - metal_temp_norm},  # metal_temp > metal_temp_norm
        {'type': 'ineq', 'fun': lambda x, ni_met_norm=ni_met_norm: ridge_model.predict(x.reshape(1, -1))[0][1] - ni_met_norm},    # Ni_met > ni_met_norm
        {'type': 'ineq', 'fun': lambda x, c_met_low_norm=c_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][2] - c_met_low_norm},  # c_met > c_met_low_norm
        {'type': 'ineq', 'fun': lambda x, c_met_high_norm=c_met_high_norm: c_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][2]},  # c_met < c_met_high_norm
        {'type': 'ineq', 'fun': lambda x, si_met_low_norm=si_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][3] - si_met_low_norm},  # si_met > si_met_low_norm
        {'type': 'ineq', 'fun': lambda x, si_met_high_norm=si_met_high_norm: si_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][3]},  # si_met < si_met_high_norm
        {'type': 'ineq', 'fun': lambda x, fe_met_low_norm=fe_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][4] - fe_met_low_norm},  # Fe_met > fe_met_low_norm
        {'type': 'ineq', 'fun': lambda x, s_met_low_norm=s_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][5] - s_met_low_norm},  # S_met > s_met_low_norm
        {'type': 'ineq', 'fun': lambda x, s_met_high_norm=s_met_high_norm: s_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][5]},  # S_met < s_met_high_norm
        {'type': 'ineq', 'fun': lambda x, ni_slag_low_norm=ni_slag_low_norm: ridge_model.predict(x.reshape(1, -1))[0][6] - ni_slag_low_norm},  # Ni_slag > ni_slag_low_norm
        {'type': 'ineq', 'fun': lambda x, fe_slag_low_norm=fe_slag_low_norm: ridge_model.predict(x.reshape(1, -1))[0][7] - fe_slag_low_norm},  # Fe_slag > fe_slag_low_norm
        {'type': 'ineq', 'fun': lambda x, t_kalsin_low_norm=t_kalsin_low_norm: ridge_model.predict(x.reshape(1, -1))[0][8] - t_kalsin_low_norm},    # t_kalsin > t_kalsin_low_norm
        {'type': 'ineq', 'fun': lambda x, pic_161_low_norm=pic_161_low_norm: ridge_model.predict(x.reshape(1, -1))[0][9] - pic_161_low_norm},  # pic_161 > pic_161_low_norm
        {'type': 'ineq', 'fun': lambda x, loi_kalsin_high=loi_kalsin_high: loi_kalsin_high - ridge_model.predict(x.reshape(1, -1))[0][10]},  # loi_kalsin < loi_kalsin_high
        {'type': 'ineq', 'fun': lambda x, loi_kalsin_low=loi_kalsin_low: ridge_model.predict(x.reshape(1, -1))[0][10] - loi_kalsin_low },  # loi_kalsin > loi_kalsin_low
    ]

    # Melakukan optimisasi menggunakan metode SLSQP dengan constraints yang didefinisikan
    result = minimize(
        objective_function,
        x0,
        method='SLSQP',
        constraints=constraints_sample
    )

    # Memeriksa apakah optimisasi berhasil
    if result.success:
        optimized_x = result.x
        optimized_y = ridge_model.predict(optimized_x.reshape(1, -1))[0]

        # Menyimpan input dan output yang dioptimalkan
        optimized_x_list.append(optimized_x)
        optimized_y_list.append(optimized_y)
    else:
        print(f"Optimisasi gagal untuk sampel {i+1}: {result.message}")
        # Menambahkan NaN jika optimisasi gagal
        optimized_x_list.append([np.nan] * len(input_cols))
        optimized_y_list.append([np.nan] * len(output_cols))

# Mengubah list input yang dioptimalkan menjadi DataFrame
optimized_x_df = pd.DataFrame(optimized_x_list, columns=input_cols)

# Inverse transform input yang dioptimalkan ke skala asli
optimized_x_df_original = pd.DataFrame(
    scaler_x.inverse_transform(optimized_x_df.fillna(0)),
    columns=input_cols
)

# Mengubah list output yang dioptimalkan menjadi DataFrame
optimized_y_df = pd.DataFrame(optimized_y_list, columns=output_cols)

# Inverse transform output yang dioptimalkan ke skala asli
optimized_y_df_original = pd.DataFrame(
    scaler_y.inverse_transform(optimized_y_df.fillna(0)),
    columns=output_cols
)

# Menampilkan hasil optimisasi untuk setiap sampel
for i in range(num_samples):
    print(f"\nSampel {i+1}:")
    if not optimized_x_df_original.iloc[i].isnull().any():
        print("Input yang Dioptimalkan:", optimized_x_df_original.iloc[i].values)
        print("Output yang Dioptimalkan:", optimized_y_df_original.iloc[i].values)
    else:
        print("Optimisasi tidak berhasil untuk sampel ini.")

# Fungsi untuk menghasilkan nama file unik dengan suffix numerik
def generate_unique_filename(base_name, extension):
    filename = f"{base_name}.{extension}"
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_name}_{counter}.{extension}"
        counter += 1
    return filename

# Menentukan nama file dasar dan ekstensi
base_filename = '../optimisasi/Optimisasi SLSQP-E'
file_extension = 'xlsx'

# Menghasilkan nama file unik
unique_filename = generate_unique_filename(base_filename, file_extension)

# Mengekspor hasil yang dioptimalkan ke file Excet l dengan nama unik
with pd.ExcelWriter(unique_filename) as writer:
    optimized_x_df_original.to_excel(writer, sheet_name='Input_Dioptimalkan', index=False)
    optimized_y_df_original.to_excel(writer, sheet_name='Output_Dioptimalkan', index=False)
    df_test[input_cols].to_excel(writer, sheet_name='Input_actual', index=False)
    df_test[output_cols].to_excel(writer, sheet_name='Output_actual', index=False)

print(f"\nHasil optimisasi telah diekspor ke file '{unique_filename}'.")

