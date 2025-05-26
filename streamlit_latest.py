import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from scipy.optimize import minimize
import os

st.set_page_config(page_title="Optimasi Proses Rotary Kiln dan Electric Furnace", layout="wide")

st.title("Optimasi Proses Rotary Kiln dan Electric Furnace")

st.write("""
Aplikasi ini digunakan untuk **menghitung konfigurasi input yang diperlukan** dalam proses **rotary kiln** dan **electric furnace** 
berdasarkan **komposisi bahan mentah** yang dimasukkan oleh pengguna. 
         
Untuk menggunakan aplikasi ini, silakan masukkan komposisi bahan mentah yang diinginkan pada kolom input di bawah ini.
""")

data_dir = "dataframe"
df_filtered = pd.read_csv(f"{data_dir}/df_filtered.csv")
df_to_optimize = pd.read_csv(f"{data_dir}/df_to_optimize.csv")

input_cols = df_to_optimize.loc[:, 'ni_in':'t_tic163'].columns
output_cols = df_to_optimize.loc[:, 'metal_temp':'loi_kalsin'].columns

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(df_filtered[input_cols])
scaler_y.fit(df_filtered[output_cols])

model_path = "ridge_model_latest.pkl"
ridge_model = load(model_path)

def inverse_transform_row(row, scaler, columns):
    arr = np.array(row).reshape(1, -1)
    return pd.Series(scaler.inverse_transform(arr)[0], index=columns)

iqr_df = df_to_optimize.quantile(0.75) - df_to_optimize.quantile(0.25)
iqr_df_for_test = iqr_df.to_frame(name='IQR').reset_index()
iqr_df_for_test.rename(columns={'index': 'Column'}, inplace=True)

std_df = df_to_optimize.std()
std_df_for_test = std_df.to_frame(name='Standard Deviation').reset_index()
std_df_for_test.rename(columns={'index': 'Column'}, inplace=True)

min_values_y = scaler_y.data_min_
max_values_y = scaler_y.data_max_
bounds_y = [(i, j) for i, j in zip(min_values_y, max_values_y)]

metal_temp = 1300
metal_temp_norm = (metal_temp - bounds_y[0][0]) / (bounds_y[0][1] - bounds_y[0][0])

ni_met = 17 - 3 * std_df_for_test.loc[std_df_for_test['Column'] == 'ni_met', 'Standard Deviation'].values[0]
ni_met_norm = (ni_met - bounds_y[1][0]) / (bounds_y[1][1] - bounds_y[1][0])

c_met_low = 1
c_met_low_norm = (c_met_low - bounds_y[2][0]) / (bounds_y[2][1] - bounds_y[2][0])
c_met_high = 3
c_met_high_norm = (c_met_high - bounds_y[2][0]) / (bounds_y[2][1] - bounds_y[2][0])

si_met_low = 1 - 1.5 * iqr_df_for_test.loc[iqr_df_for_test['Column'] == 'si_met', 'IQR'].values[0]
si_met_low_norm = (si_met_low - bounds_y[3][0]) / (bounds_y[3][1] - bounds_y[3][0])
si_met_high = 2
si_met_high_norm = (si_met_high - bounds_y[3][0]) / (bounds_y[3][1] - bounds_y[3][0])

fe_met_low = 0
fe_met_low_norm = (fe_met_low - bounds_y[4][0]) / (bounds_y[4][1] - bounds_y[4][0])

s_met_low = 0
s_met_low_norm = (s_met_low - bounds_y[5][0]) / (bounds_y[5][1] - bounds_y[5][0])
s_met_high = 0.4 + 1.5 * iqr_df_for_test.loc[iqr_df_for_test['Column'] == 's_met', 'IQR'].values[0]
s_met_high_norm = (s_met_high - bounds_y[5][0]) / (bounds_y[5][1] - bounds_y[5][0])

ni_slag_low = 0
ni_slag_low_norm = (ni_slag_low - bounds_y[6][0]) / (bounds_y[6][1] - bounds_y[6][0])

fe_slag_low = 0
fe_slag_low_norm = (fe_slag_low - bounds_y[7][0]) / (bounds_y[7][1] - bounds_y[7][0])

t_kalsin_low = 600
t_kalsin_low_norm = (t_kalsin_low - bounds_y[8][0]) / (bounds_y[8][1] - bounds_y[8][0])

pic_161_low = -5.28
pic_161_low_norm = (pic_161_low - bounds_y[9][0]) / (bounds_y[9][1] - bounds_y[9][0])

loi_kalsin_low = 0
loi_kalsin_low_norm = (loi_kalsin_low - bounds_y[10][0]) / (bounds_y[10][1] - bounds_y[10][0])
loi_kalsin_high = 1
loi_kalsin_high_norm = (loi_kalsin_high - bounds_y[10][0]) / (bounds_y[10][1] - bounds_y[10][0])

min_values_x = scaler_x.data_min_
max_values_x = scaler_x.data_max_
bounds_x = [(i, j) for i, j in zip(min_values_x, max_values_x)]

current_pry_min = 0
current_pry_min_norm = (current_pry_min - bounds_x[11][0]) / (bounds_x[11][1] - bounds_x[11][0])

current_sec1_min = 0
current_sec1_min_norm = (current_sec1_min - bounds_x[12][0]) / (bounds_x[12][1] - bounds_x[12][0])

current_sec2_min = 0
current_sec2_min_norm = (current_sec2_min - bounds_x[13][0]) / (bounds_x[13][1] - bounds_x[13][0])

current_sec3_min = 0
current_sec3_min_norm = (current_sec3_min - bounds_x[14][0]) / (bounds_x[14][1] - bounds_x[14][0])

load_min = 0
load_min_norm = (load_min - bounds_x[15][0]) / (bounds_x[15][1] - bounds_x[15][0])

realisasi_beban_min = 0
realisasi_beban_min_norm = (realisasi_beban_min - bounds_x[16][0]) / (bounds_x[16][1] - bounds_x[16][0])

rpm_min = 0
rpm_min_norm = (rpm_min - bounds_x[17][0]) / (bounds_x[17][1] - bounds_x[17][0])
rpm_max = np.percentile(df_to_optimize['rpm'], 50)
rpm_max_norm = (rpm_max - bounds_x[17][0]) / (bounds_x[17][1] - bounds_x[17][0])

pry_p_min = 0
pry_p_min_norm = (pry_p_min - bounds_x[21][0]) / (bounds_x[21][1] - bounds_x[21][0])
pry_p_max = np.percentile(df_to_optimize['pry_p'], 50)
pry_p_max_norm = (pry_p_max - bounds_x[21][0]) / (bounds_x[21][1] - bounds_x[21][0])

sec_p_min = 0
sec_p_min_norm = (sec_p_min - bounds_x[22][0]) / (bounds_x[22][1] - bounds_x[22][0])
sec_p_max = np.percentile(df_to_optimize['sec_p'], 50)
sec_p_max_norm = (sec_p_max - bounds_x[22][0]) / (bounds_x[22][1] - bounds_x[22][0])

pry_v_min = 0
pry_v_min_norm = (pry_v_min - bounds_x[23][0]) / (bounds_x[23][1] - bounds_x[23][0])
pry_v_max = np.percentile(df_to_optimize['pry_v'], 50)
pry_v_max_norm = (pry_v_max - bounds_x[23][0]) / (bounds_x[23][1] - bounds_x[23][0])

sec_v_min = 0
sec_v_min_norm = (sec_v_min - bounds_x[24][0]) / (bounds_x[24][1] - bounds_x[24][0])
sec_v_max = np.percentile(df_to_optimize['sec_v'], 50)
sec_v_max_norm = (sec_v_max - bounds_x[24][0]) / (bounds_x[24][1] - bounds_x[24][0])

total_fuel_min = 0
total_fuel_min_norm = (total_fuel_min - bounds_x[25][0]) / (bounds_x[25][1] - bounds_x[25][0])
total_fuel_max = np.percentile(df_to_optimize['total_fuel'], 50)
total_fuel_max_norm = (total_fuel_max - bounds_x[25][0]) / (bounds_x[25][1] - bounds_x[25][0])

reductor_consume_min = 0
reductor_consume_min_norm = (reductor_consume_min - bounds_x[26][0]) / (bounds_x[26][1] - bounds_x[26][0])
reductor_consume_max = np.percentile(df_to_optimize['reductor_consume'], 50)
reductor_consume_max_norm = (reductor_consume_max - bounds_x[26][0]) / (bounds_x[26][1] - bounds_x[26][0])

t_tic162_min = 556
t_tic162_min_norm = (t_tic162_min - bounds_x[27][0]) / (bounds_x[27][1] - bounds_x[27][0])
t_tic162_max = np.percentile(df_to_optimize['t_tic162'], 50)
t_tic162_max_norm = (t_tic162_max - bounds_x[27][0]) / (bounds_x[27][1] - bounds_x[27][0])

t_tic163_min = 456
t_tic163_min_norm = (t_tic163_min - bounds_x[28][0]) / (bounds_x[28][1] - bounds_x[28][0])
t_tic163_max = 845
t_tic163_max_norm = (t_tic163_max - bounds_x[28][0]) / (bounds_x[28][1] - bounds_x[28][0])

input_ni = st.number_input("Ni (%):", min_value=0.0, format="%.2f")
input_fe = st.number_input("Fe (%):", min_value=0.0, format="%.2f")
input_cao = st.number_input("CaO (%):", min_value=0.0, format="%.2f")
input_al2o3 = st.number_input("Al2O3 (%):", min_value=0.0, format="%.2f")
input_s_m = st.number_input("S/M Ratio (%):", min_value=0.0, format="%.2f")
input_bc = st.number_input("BC:", min_value=0.0, format="%.2f")
input_mc_kilnfeed = st.number_input("MC Kilnfeed (%):", min_value=0.0, format="%.2f")
input_fc_coal = st.number_input("FC Coal (%):", min_value=0.0, format="%.2f")
input_gcv_coal = st.number_input("GCV Coal (kcal/kg):", min_value=0.0, format="%.2f")
input_fc_lcv = st.number_input("FC LCV (%):", min_value=0.0, format="%.2f")
input_gcv_lcv = st.number_input("GCV LCV (kcal/kg):", min_value=0.0, format="%.2f")
input_kg_tco = st.number_input("KG TCO (ton/jam):", min_value=0.0, format="%.2f")
input_charge_kiln = st.number_input("Charge Kiln (ton/jam):", min_value=0.0, format="%.2f")
input_tdo = st.number_input("TDO (ton/jam):", min_value=0.0, format="%.2f")

def optimize_sample(fixed_inputs):
    x0 = scaler_x.transform([fixed_inputs])[0]
    eq_vals = x0.copy()
    
    def objective(x):
        y_pred = ridge_model.predict(x.reshape(1, -1))[0]
        return np.sum(y_pred - ridge_model.predict(x0.reshape(1, -1))[0]) ** 2
    
    constraints = [
        {'type': 'eq', 'fun': lambda x, i=i, v=eq_vals[i]: x[i] - v} for i in range(0, 6)
    ]
    constraints += [
        {'type': 'eq', 'fun': lambda x, i=i, v=eq_vals[i]: x[i] - v} for i in [6,7,8,9,10,18,19,20]
    ]
    constraints += [
        {'type': 'ineq', 'fun': lambda x: x[11] - current_pry_min_norm},
        {'type': 'ineq', 'fun': lambda x: x[12] - current_sec1_min_norm},
        {'type': 'ineq', 'fun': lambda x: x[13] - current_sec2_min_norm},
        {'type': 'ineq', 'fun': lambda x: x[14] - current_sec3_min_norm},
        {'type': 'ineq', 'fun': lambda x: x[15] - load_min_norm},
        {'type': 'ineq', 'fun': lambda x: x[16] - realisasi_beban_min_norm},
        {'type': 'ineq', 'fun': lambda x: x[17] - rpm_min_norm},
        {'type': 'ineq', 'fun': lambda x: rpm_max_norm - x[17]},
        {'type': 'ineq', 'fun': lambda x: x[21] - pry_p_min_norm},
        {'type': 'ineq', 'fun': lambda x: pry_p_max_norm - x[21]},
        {'type': 'ineq', 'fun': lambda x: x[22] - sec_p_min_norm},
        {'type': 'ineq', 'fun': lambda x: sec_p_max_norm - x[22]},
        {'type': 'ineq', 'fun': lambda x: x[23] - pry_v_min_norm},
        {'type': 'ineq', 'fun': lambda x: pry_v_max_norm - x[23]},
        {'type': 'ineq', 'fun': lambda x: x[24] - sec_v_min_norm},
        {'type': 'ineq', 'fun': lambda x: sec_v_max_norm - x[24]},
        {'type': 'ineq', 'fun': lambda x: x[25] - total_fuel_min_norm},
        {'type': 'ineq', 'fun': lambda x: total_fuel_max_norm - x[25]},
        {'type': 'ineq', 'fun': lambda x: x[26] - reductor_consume_min_norm},
        {'type': 'ineq', 'fun': lambda x: reductor_consume_max_norm - x[26]},
        {'type': 'ineq', 'fun': lambda x: x[27] - t_tic162_min_norm},
        {'type': 'ineq', 'fun': lambda x: t_tic162_max_norm - x[27]},
        {'type': 'ineq', 'fun': lambda x: x[28] - t_tic163_min_norm},
        {'type': 'ineq', 'fun': lambda x: t_tic163_max_norm - x[28]},
        {'type': 'ineq', 'fun': lambda x, metal_temp_norm=metal_temp_norm: ridge_model.predict(x.reshape(1, -1))[0][0] - metal_temp_norm},
        {'type': 'ineq', 'fun': lambda x, ni_met_norm=ni_met_norm: ridge_model.predict(x.reshape(1, -1))[0][1] - ni_met_norm},
        {'type': 'ineq', 'fun': lambda x, c_met_low_norm=c_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][2] - c_met_low_norm},
        {'type': 'ineq', 'fun': lambda x, c_met_high_norm=c_met_high_norm: c_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][2]},
        {'type': 'ineq', 'fun': lambda x, si_met_low_norm=si_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][3] - si_met_low_norm},
        {'type': 'ineq', 'fun': lambda x, si_met_high_norm=si_met_high_norm: si_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][3]},
        {'type': 'ineq', 'fun': lambda x, fe_met_low_norm=fe_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][4] - fe_met_low_norm},
        {'type': 'ineq', 'fun': lambda x, s_met_low_norm=s_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][5] - s_met_low_norm},
        {'type': 'ineq', 'fun': lambda x, s_met_high_norm=s_met_high_norm: s_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][5]},
        {'type': 'ineq', 'fun': lambda x, ni_slag_low_norm=ni_slag_low_norm: ridge_model.predict(x.reshape(1, -1))[0][6] - ni_slag_low_norm},
        {'type': 'ineq', 'fun': lambda x, fe_slag_low_norm=fe_slag_low_norm: ridge_model.predict(x.reshape(1, -1))[0][7] - fe_slag_low_norm},
        {'type': 'ineq', 'fun': lambda x, t_kalsin_low_norm=t_kalsin_low_norm: ridge_model.predict(x.reshape(1, -1))[0][8] - t_kalsin_low_norm},
        {'type': 'ineq', 'fun': lambda x, pic_161_low_norm=pic_161_low_norm: ridge_model.predict(x.reshape(1, -1))[0][9] - pic_161_low_norm},
        {'type': 'ineq', 'fun': lambda x, loi_kalsin_high_norm=loi_kalsin_high_norm: loi_kalsin_high_norm - ridge_model.predict(x.reshape(1, -1))[0][10]},
        {'type': 'ineq', 'fun': lambda x, loi_kalsin_low_norm=loi_kalsin_low_norm: ridge_model.predict(x.reshape(1, -1))[0][10] - loi_kalsin_low_norm},
    ]
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints
    )
    
    return result

def create_input_array():
    input_dict = {
        'ni_in': input_ni,
        'fe_in': input_fe,
        'cao_in': input_cao,
        'al2o3_in': input_al2o3,
        's_m': input_s_m,
        'bc': input_bc,
        'mc_kilnfeed': input_mc_kilnfeed,
        'fc_coal': input_fc_coal,
        'gcv_coal': input_gcv_coal,
        'fc_lcv': input_fc_lcv,
        'gcv_lcv': input_gcv_lcv,
        'kg_tco': input_kg_tco,
        'charge_kiln': input_charge_kiln, 
        'tdo': input_tdo
    }

    input_array = []
    for col in input_cols:
        if col in input_dict:
            input_array.append(input_dict[col])
        else:
            input_array.append(df_filtered[col].median())
    
    return np.array(input_array)

button_hitung = st.button("Hitung", key="hitung_button")

if button_hitung:
    with st.spinner("Mengoptimasi... Mohon tunggu."):
        user_input_arr = create_input_array()
        
        result = optimize_sample(user_input_arr)
        
        if result.success:
            optimized_x = result.x
            optimized_y = ridge_model.predict(optimized_x.reshape(1, -1))[0]
            
            optimized_x_orig = inverse_transform_row(optimized_x, scaler_x, input_cols)
            optimized_y_orig = inverse_transform_row(optimized_y, scaler_y, output_cols)
            
            output = {}
            
            column_mapping = {
                "Tegangan (Volt)": "voltage" if "voltage" in optimized_x_orig else None,
                "Arus (Kilo Ampere)": "current_pry" if "current_pry" in optimized_x_orig else None,
                "Beban Listrik (Load MW)": "load" if "load" in optimized_x_orig else None,
                "Kecepatan Putar (RPM)": "rpm" if "rpm" in optimized_x_orig else None,
                "Aliran Udara Katup Primer (Pry_p)": "pry_p" if "pry_p" in optimized_x_orig else None,
                "Aliran Udara Katup Sekunder (Sec_p)": "sec_p" if "sec_p" in optimized_x_orig else None,
                "Tekanan Udara Primer (Pry_v)": "pry_v" if "pry_v" in optimized_x_orig else None,
                "Tekanan Udara Sekunder (Sec_v)": "sec_v" if "sec_v" in optimized_x_orig else None,
                "Total Konsumsi Batu Bara (Total Fuel)": "total_fuel" if "total_fuel" in optimized_x_orig else None,
                "Rasio Udara-Bahan Bakar (A/F Ratio)": "a_f_ratio" if "a_f_ratio" in optimized_x_orig else None,
                "Rasio Reduktor": "reductor_ratio" if "reductor_ratio" in optimized_x_orig else None,
                "Konsumsi Reduktor (Reductor Consume)": "reductor_consume" if "reductor_consume" in optimized_x_orig else None,
                "Temperatur TIC 162": "t_tic162" if "t_tic162" in optimized_x_orig else None,
                "Temperatur TIC 163": "t_tic163" if "t_tic163" in optimized_x_orig else None
            }
            
            for display_name, col_name in column_mapping.items():
                if col_name and col_name in optimized_x_orig:
                    output[display_name] = round(optimized_x_orig[col_name], 3)
            
            output_y = {}
            y_column_mapping = {
                "Temperatur Metal (Temp Metal)": "metal_temp" if "metal_temp" in optimized_y_orig else None,
                "Kandungan Ni di Metal (Ni Met)": "ni_met" if "ni_met" in optimized_y_orig else None,
                "Kandungan C di Metal (C Met)": "c_met" if "c_met" in optimized_y_orig else None,
                "Kandungan Si di Metal (Si Met)": "si_met" if "si_met" in optimized_y_orig else None,
                "Kandungan Fe di Metal (Fe Met)": "fe_met" if "fe_met" in optimized_y_orig else None,
                "Kandungan S di Metal (S Met)": "s_met" if "s_met" in optimized_y_orig else None,
                "Kandungan Ni di Slag (Ni Slag)": "ni_slag" if "ni_slag" in optimized_y_orig else None,
                "Kandungan Fe di Slag (Fe Slag)": "fe_slag" if "fe_slag" in optimized_y_orig else None,
                "Temperatur Kalsinasi (T Kalsin)": "t_kalsin" if "t_kalsin" in optimized_y_orig else None,
                "Temperatur PIC 161": "pic_161" if "pic_161" in optimized_y_orig else None,
                "Loss on Ignition Kalsinasi (LOI Kalsin)": "loi_kalsin" if "loi_kalsin" in optimized_y_orig else None
            }
            
            for display_name, col_name in y_column_mapping.items():
                if col_name and col_name in optimized_y_orig:
                    output_y[display_name] = round(optimized_y_orig[col_name], 3)

            output_html_1 = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"
            output_html_1 += "<tr><th style='padding: 8px; text-align: left;'>Parameter</th><th style='padding: 8px; text-align: left;'>Value</th></tr>"

            for param, value in output.items():
                formatted_value = f"{value:.2f}".replace('.', ',')
                output_html_1 += f"<tr><td style='padding: 8px; border: 1px solid black;'>{param}</td><td style='padding: 8px; border: 1px solid black;'>{formatted_value}</td></tr>"

            output_html_1 += "</table>"

            output_html_2 = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"
            output_html_2 += "<tr><th style='padding: 8px; text-align: left;'>Parameter</th><th style='padding: 8px; text-align: left;'>Value</th></tr>"

            for param, value in output_y.items():
                formatted_value = f"{value:.2f}".replace('.', ',')
                output_html_2 += f"<tr><td style='padding: 8px; border: 1px solid black;'>{param}</td><td style='padding: 8px; border: 1px solid black;'>{formatted_value}</td></tr>"

            output_html_2 += "</table>"

            st.markdown("### Input", unsafe_allow_html=True)
            st.markdown(output_html_1, unsafe_allow_html=True)

            st.markdown("### Output Target", unsafe_allow_html=True)
            st.markdown(output_html_2, unsafe_allow_html=True)
            
        else:
            st.error(f"Optimisasi gagal: {result.message}")