import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.optimize import minimize
import pickle
from joblib import load
import os
import warnings

# Menampilkan judul aplikasi
st.title("Optimasi Proses Rotary Kiln dan Electric Furnace")

# Deskripsi aplikasi
st.write("""
Aplikasi ini digunakan untuk **menghitung konfigurasi input yang diperlukan** dalam proses **rotary kiln** dan **electric furnace** 
berdasarkan **komposisi bahan mentah** yang dimasukkan oleh pengguna. 
         
Untuk menggunakan aplikasi ini, silakan masukkan komposisi bahan mentah yang diinginkan pada kolom input di bawah ini(value placeholder dapat diubah).
""")

# Kolom Input dari pengguna dengan teks diperbarui dan tanpa nilai default
input_ni = st.number_input("Ni (ppm):", min_value=0.0, format="%.2f")
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
input_charge_klin = st.number_input("Charge Kiln (ton/jam):", min_value=0.0, format="%.2f")
input_tdo = st.number_input("TDO (ton/jam):", min_value=0.0, format="%.2f")

# Tombol untuk mengirim data
if st.button("Hitung"):

    def load_data():
        pd.options.display.max_columns = None
        df = pd.read_csv("df_to_optimize_actual.csv", encoding="unicode_escape")

        ridge_model = load("ridge_regression_model.pkl")
        df = df.drop(columns=['total_coal'])
        return df, ridge_model

    def scale_data(df, input_cols, output_cols):
        input_cols = df.columns[df.columns.get_loc('ni_in'):df.columns.get_loc('t_tic163') + 1]
        output_cols = df.columns[df.columns.get_loc('metal_temp'):df.columns.get_loc('loi_kalsin') + 1]

        scalers = {}
        scaled_data = {}

        for col in df:
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(df[[col]])[:, 0]
            scalers[col] = scaler

        norm_df = pd.DataFrame(scaled_data)
        return norm_df, scalers, input_cols, output_cols

    def define_constraints(df, norm_df, scalers, input_cols, output_cols, initial_cond_norm_df):
        constraint_output_min = {
            "metal_temp": 1300,
            "ni_met": 17,
            "c_met": 0,
            "si_met": 1,
            "fe_met": 0,
            "s_met": 0,
            "ni_slag": 0,
            "fe_slag": 0,
            "t_kalsin": 600,
            "pic_161": -1,
            "loi_kalsin": 0
        }

        constraint_output_max = {
            "si_met": 2,
            "s_met": 0.5,
            "loi_kalsin": 1
        }

        constraint_output_min_df = pd.DataFrame([constraint_output_min])
        constraint_output_max_df = pd.DataFrame([constraint_output_max])

        constraint_output_min_norm_df = pd.DataFrame({col: scalers[col].transform(constraint_output_min_df[[col]])[:, 0] for col in constraint_output_min_df.columns})
        constraint_output_max_norm_df = pd.DataFrame({col: scalers[col].transform(constraint_output_max_df[[col]])[:, 0] for col in constraint_output_max_df.columns})

        constraint_input_min = {
            "ni_in": 0,
            "fe_in": 0,
            "cao_in": 0,
            "al2o3_in": 0,
            "s_m": 0,
            "bc": 0,
            "current_pry": 0,
            "current_sec1": 0,
            "current_sec2": 0,
            "current_sec3": 0,
            "load": 0,
            "power_factor": 0,
            "realisasi_beban": 0,
            "rpm": 0,
            "pry_p": 0,
            "sec_p": 0,
            "pry_v": 0,
            "sec_v": 0,
            "total_fuel": 0,
            "a_f_ratio": 0,
            "reductor_consume": 1,
            "t_tic162": 556,
            "t_tic163": 456
        }

        constraint_input_max = {
            "t_tic162": 1201,
            "t_tic163": 845
        }

        constraint_input_fixed = {
            "mc_kilnfeed": initial_cond_norm_df['mc_kilnfeed'],
            "fc_coal": initial_cond_norm_df['fc_coal'],
            "gcv_coal": initial_cond_norm_df['gcv_coal'],
            "fc_lcv": initial_cond_norm_df['fc_lcv'],
            "gcv_lcv": initial_cond_norm_df['gcv_lcv'],
            "kg_tco": initial_cond_norm_df['kg_tco'],
            "charge_kiln": initial_cond_norm_df['charge_kiln'],
            "tdo": initial_cond_norm_df['tdo']
        }

        constraint_input_min_df = pd.DataFrame([constraint_input_min])
        constraint_input_max_df = pd.DataFrame([constraint_input_max])
        constraint_input_fixed_norm_df = pd.DataFrame([constraint_input_fixed])

        constraint_input_min_norm_df = pd.DataFrame({col: scalers[col].transform(constraint_input_min_df[[col]])[:, 0] for col in constraint_input_min_df.columns})
        constraint_input_max_norm_df = pd.DataFrame({col: scalers[col].transform(constraint_input_max_df[[col]])[:, 0] for col in constraint_input_max_df.columns})

        return constraint_input_min_norm_df, constraint_input_max_norm_df, constraint_output_min_norm_df, constraint_output_max_norm_df, constraint_input_fixed_norm_df

    def objective_function(x_scaled, ridge_model, target_value):
        x_scaled = x_scaled.reshape(1, -1)
        y_pred = ridge_model.predict(x_scaled)[0]
        return (y_pred[0] - target_value[0]) ** 2

    def optimize_row(df, norm_df, scalers, input_cols, output_cols, ridge_model, row_idx):
        initial_cond_df = df.iloc[[row_idx]]
        initial_cond_x_df = initial_cond_df[input_cols]
        initial_cond_y_df = initial_cond_df[output_cols]

        initial_cond_norm_df = norm_df.loc[[row_idx]]
        initial_cond_norm_x_df = initial_cond_norm_df[input_cols]
        initial_cond_norm_y_df = initial_cond_norm_df[output_cols]

        target_value = initial_cond_norm_y_df.iloc[0, :].to_numpy()

        constraint_input_min_norm_df, constraint_input_max_norm_df, constraint_output_min_norm_df, constraint_output_max_norm_df, constraint_input_fixed_norm_df = define_constraints(df, norm_df, scalers, input_cols, output_cols, initial_cond_norm_df)

        x0 = initial_cond_norm_x_df.to_numpy().flatten()
        constraints = []

        for col in constraint_input_min_norm_df.columns:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, col=col: x[input_cols.get_loc(col)] - constraint_input_min_norm_df[col][0]
            })

        for col in constraint_input_max_norm_df.columns:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, col=col: constraint_input_max_norm_df[col][0] - x[input_cols.get_loc(col)]
            })

        constraints.append({
            'type': 'ineq', 
            'fun': lambda x: initial_cond_norm_df['total_fuel'].values[0] - x[input_cols.get_loc('total_fuel')]
        })

        constraints.append({
            'type': 'ineq', 
            'fun': lambda x: initial_cond_norm_df['reductor_consume'].values[0] - x[input_cols.get_loc('reductor_consume')]
        })

        for col in constraint_output_min_norm_df.columns:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, col=col: x[output_cols.get_loc(col)] - constraint_output_min_norm_df[col][0]
            })

        for col in constraint_output_max_norm_df.columns:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, col=col: constraint_output_max_norm_df[col][0] - x[output_cols.get_loc(col)]
            })

        for col in constraint_input_fixed_norm_df.columns:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x, col=col: constraint_input_fixed_norm_df[col][0] - x[input_cols.get_loc(col)]
            })

        result = minimize(objective_function, x0, args=(ridge_model, target_value), method='SLSQP', constraints=constraints)

        optimized_x = result.x
        optimized_y = ridge_model.predict(optimized_x.reshape(1, -1))[0]

        optimized_x_df = pd.DataFrame(optimized_x.reshape(1, -1), columns=input_cols)
        optimized_y_df = pd.DataFrame(optimized_y.reshape(1, -1), columns=output_cols)

        optimized_x_inv = pd.DataFrame({col: scalers[col].inverse_transform(optimized_x_df[[col]])[:, 0] for col in optimized_x_df.columns})
        optimized_y_inv = pd.DataFrame({col: scalers[col].inverse_transform(optimized_y_df[[col]])[:, 0] for col in optimized_y_df.columns})

        return optimized_x_inv, optimized_y_inv, initial_cond_x_df, initial_cond_y_df

    def main(num_rows_to_optimize):
        df, ridge_model = load_data()

        input_cols = df.columns[df.columns.get_loc('ni_in'):df.columns.get_loc('t_tic163') + 1]
        output_cols = df.columns[df.columns.get_loc('metal_temp'):df.columns.get_loc('loi_kalsin') + 1]

        # Input pengguna (15 nilai yang tidak dioptimalkan, dikunci)
        user_input = {
            'ni_in': input_ni,
            'fe_in': input_fe,
            'cao_in': input_cao,
            'al2o3_in': input_al2o3,
            's_m': input_s_m,
            'bc': input_bc,
            'mc_kilnfeed': input_mc_kilnfeed,
            'fc_coal': input_fc_coal,
            'gcv_coal': input_gcv_coal,
            'charge_kiln': input_charge_klin,
        }

        norm_df, scalers, input_cols, output_cols = scale_data(df, input_cols, output_cols)

        optimized_rows = []

        error_metrics = []

        for i in range(num_rows_to_optimize):
            optimized_x_inv, optimized_y_inv, initial_cond_x_df, initial_cond_y_df = optimize_row(df, norm_df, scalers, input_cols, output_cols, ridge_model, i)

            optimized_row = optimized_x_inv.copy()
            optimized_row[output_cols] = optimized_y_inv
            optimized_rows.append(optimized_row)

            print(f"\n Optimizing Row {i} ")

            # Calculate error metrics for each column
            for col in initial_cond_x_df.columns:
                mae = mean_absolute_error(initial_cond_x_df[[col]], optimized_x_inv[[col]])
                mse = mean_squared_error(initial_cond_x_df[[col]], optimized_x_inv[[col]])
                rmse = np.sqrt(mse)

                error_metrics.append({
                'row_index': i,
                'column': col,
                'actual_value': initial_cond_x_df[col].values[0],
                'optimized_value': optimized_x_inv[col].values[0],
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
                })

            # Insert a blank row to separate each index
            error_metrics.append({
                'row_index': '',
                'column': '',
                'actual_value': '',
                'MAE': '',
                'MSE': '',
                'RMSE': ''
            })

            error_metrics_df = pd.DataFrame(error_metrics)

        optimized_df = pd.concat(optimized_rows, ignore_index=True)
        optimized_df.insert(0, 'row_index', range(num_rows_to_optimize))

        comparison_df_reductor = pd.DataFrame({
            'row_index': range(num_rows_to_optimize),
            'actual_reductor_consume': df['reductor_consume'][:num_rows_to_optimize].values,
            'optimized_reductor_consume': optimized_df['reductor_consume'].values,
        })

        comparison_df_reductor['AE'] = abs(comparison_df_reductor['actual_reductor_consume'] - comparison_df_reductor['optimized_reductor_consume'])
        comparison_df_reductor['APE'] = (comparison_df_reductor['AE'] / comparison_df_reductor['actual_reductor_consume']) * 100
        comparison_df_reductor['Efficiency'] = comparison_df_reductor['actual_reductor_consume'] - comparison_df_reductor['optimized_reductor_consume']

        comparison_df_fuel = pd.DataFrame({
            'row_index': range(num_rows_to_optimize),
            'actual_total_fuel': df['total_fuel'][:num_rows_to_optimize].values,
            'optimized_total_fuel': optimized_df['total_fuel'].values,
        })

        comparison_df_fuel['AE'] = abs(comparison_df_fuel['actual_total_fuel'] - comparison_df_fuel['optimized_total_fuel'])
        comparison_df_fuel['APE'] = (comparison_df_fuel['AE'] / comparison_df_fuel['actual_total_fuel']) * 100
        comparison_df_fuel['Efficiency'] = comparison_df_fuel['actual_total_fuel'] - comparison_df_fuel['optimized_total_fuel']

        print("\n===== Comparison DataFrame for Reductor Consume =====")
        print(comparison_df_reductor)

        print("\n===== Comparison DataFrame for Total Fuel =====")
        print(comparison_df_fuel)

        try:
            with pd.ExcelWriter("SLSQP_Optimization_Report.xlsx") as writer:
                error_metrics_df.to_excel(writer, sheet_name="Error Metrics", index=False, float_format="%.6f")
                comparison_df_reductor.to_excel(writer, sheet_name="Reductor Consume", index=False, float_format="%.6f")
                comparison_df_fuel.to_excel(writer, sheet_name="Total Fuel", index=False, float_format="%.6f")
        except FileNotFoundError:
            with pd.ExcelWriter("SLSQP_Optimization_Report.xlsx", mode='w') as writer:
                error_metrics_df.to_excel(writer, sheet_name="Error Metrics", index=False, float_format="%.6f")
                comparison_df_reductor.to_excel(writer, sheet_name="Reductor Consume", index=False, float_format="%.6f")
                comparison_df_fuel.to_excel(writer, sheet_name="Total Fuel", index=False, float_format="%.6f")

        return optimized_df, comparison_df_fuel, comparison_df_reductor, error_metrics_df

    warnings.simplefilter("ignore", UserWarning)
    df_interpolated = pd.read_csv("df_to_optimize_actual.csv", encoding="unicode_escape")
    num_rows_to_optimize = len(df_interpolated)
    # num_rows_to_optimize = 5
    main(num_rows_to_optimize=num_rows_to_optimize)

    

    # Urutan input yang lengkap
    input_order = [
        'ni_in', 'fe_in', 'sio2_in', 'cao_in', 'mgo_in', 'al2o3_in', 'fe_ni', 's_m', 'bc', 'loi_in',
        'mc_kilnfeed', 'fc_coal', 'gcv_coal', 'tco', 'voltage', 'current', 'load', 'rpm', 'pry_p', 'sec_p',
        'pry_v', 'sec_v', 'total_coal', 'a_f_ratio', 'kg_tco', 'reductor_ratio', 'reductor_consume', 'charge_kiln',
        't_tic162', 't_tic163', 't_tic166', 't_tic172'
    ]

    # Menyusun nilai input sesuai urutan yang benar
    full_input = []
    for col in input_order:
        if col in user_input:
            # Menambahkan input pengguna yang dikunci
            full_input.append(user_input[col])
        else:
            # Menambahkan nilai acak untuk input yang tidak diberikan oleh pengguna
            full_input.append(random_input[0])
            random_input = random_input[1:]  # Mengurangi array random_input

    # Mengonversi full_input ke dalam array numpy
    full_input = np.array(full_input)

    # Menormalisasi input menggunakan scaler yang sudah disimpan
    full_input_scaled = scaler_x.transform(full_input.reshape(1, -1))

    # Mengakses nilai-nilai khusus dari array yang dinormalisasi
    ni_in_exact_norm = full_input_scaled[0][input_order.index('ni_in')]
    fe_in_exact_norm = full_input_scaled[0][input_order.index('fe_in')]
    sio2_in_exact_norm = full_input_scaled[0][input_order.index('sio2_in')]
    cao_in_exact_norm = full_input_scaled[0][input_order.index('cao_in')]
    mgo_in_exact_norm = full_input_scaled[0][input_order.index('mgo_in')]
    al2o3_in_exact_norm = full_input_scaled[0][input_order.index('al2o3_in')]
    fe_ni_exact_norm = full_input_scaled[0][input_order.index('fe_ni')]
    s_m_exact_norm = full_input_scaled[0][input_order.index('s_m')]
    bc_exact_norm = full_input_scaled[0][input_order.index('bc')]
    loi_in_exact_norm = full_input_scaled[0][input_order.index('loi_in')]
    mc_kilnfeed_exact_norm = full_input_scaled[0][input_order.index('mc_kilnfeed')]
    fc_coal_exact_norm = full_input_scaled[0][input_order.index('fc_coal')]
    gvc_coal_exact_norm = full_input_scaled[0][input_order.index('gcv_coal')]
    tco_exact_norm = full_input_scaled[0][input_order.index('tco')]
    kg_tco_exact_norm = full_input_scaled[0][input_order.index('kg_tco')]

    optimized_x_list = []
    optimized_y_list = []

    # Melakukan iterasi melalui 10 sampel pertama di X_test
    for i in range(1):
        print(f"\nMemproses Sampel ke-{i+1}")
        # Mengambil sampel ke-i dan meratakan array-nya
        x0 = X_test.iloc[i, :].to_numpy().flatten()

        # Pastikan y_test hanya memiliki satu kolom output jika prediksi satu fitur
        target_value = y_test.iloc[i, :].to_numpy()

        # Mendefinisikan fungsi objektif spesifik untuk sampel ke-i
        def objective_function(x_scaled):
            # Mengubah bentuk input untuk prediksi
            x_scaled_reshaped = x_scaled.reshape(1, -1)
            # Melakukan prediksi menggunakan model ridge
            y_pred = ridge_model.predict(x_scaled_reshaped)[0]
            # Menghitung selisih kuadrat (MSE)
            return (y_pred[i] - target_value[i]) ** 2

        # Mendefinisikan constraints spesifik untuk sampel ke-i
        constraints_sample = [
            # Constraints equality (tidak berubah) (harusnya equality constraint, dikunci berdasarkan data asli)
            {'type': 'eq', 'fun': lambda x, ni_in_exact_norm=ni_in_exact_norm: x[0] - ni_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, fe_in_exact_norm=fe_in_exact_norm: x[1] - fe_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, sio2_in_exact_norm=sio2_in_exact_norm: x[2] - sio2_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, cao_in_exact_norm=cao_in_exact_norm: x[3] - cao_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, mgo_in_exact_norm=mgo_in_exact_norm: x[4] - mgo_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, al2o3_in_exact_norm=al2o3_in_exact_norm: x[5] - al2o3_in_exact_norm},
            {'type': 'eq', 'fun': lambda x, fe_ni_exact_norm=fe_ni_exact_norm: x[6] - fe_ni_exact_norm},
            {'type': 'eq', 'fun': lambda x, s_m_exact_norm=s_m_exact_norm: x[7] - s_m_exact_norm},
            {'type': 'eq', 'fun': lambda x, bc_exact_norm=bc_exact_norm: x[8] - bc_exact_norm},
            {'type': 'eq', 'fun': lambda x, loi_in_exact_norm=loi_in_exact_norm: x[9] - loi_in_exact_norm},

            # Constraints equality (tidak berubah)
            {'type': 'eq', 'fun': lambda x, mc_kilnfeed_exact_norm=mc_kilnfeed_exact_norm: mc_kilnfeed_exact_norm - x[10]},  # mc_kilnfeed tidak berubah
            {'type': 'eq', 'fun': lambda x, fc_coal_exact_norm=fc_coal_exact_norm: fc_coal_exact_norm - x[11]},  # fc_coal tidak berubah
            {'type': 'eq', 'fun': lambda x, gvc_coal_exact_norm=gvc_coal_exact_norm: gvc_coal_exact_norm - x[12]},  # gvc_coal tidak berubah
            {'type': 'eq', 'fun': lambda x, tco_exact_norm=tco_exact_norm: tco_exact_norm - x[13]},  # tco tidak berubah

            # Constraints input >= min_norm
            {'type': 'ineq', 'fun': lambda x, voltage_min_norm=voltage_min_norm: x[14] - voltage_min_norm},  # voltage > voltage_min_norm
            {'type': 'ineq', 'fun': lambda x, current_min_norm=current_min_norm: x[15] - current_min_norm},  # current > current_min_norm
            {'type': 'ineq', 'fun': lambda x, load_min_norm=load_min_norm: x[16] - load_min_norm},  # load > load_min_norm
            {'type': 'ineq', 'fun': lambda x, rpm_min_norm=rpm_min_norm: x[17] - rpm_min_norm},  # rpm > rpm_min_norm
            {'type': 'ineq', 'fun': lambda x, pry_p_min_norm=pry_p_min_norm: x[18] - pry_p_min_norm},  # pry_p > pry_p_min_norm
            {'type': 'ineq', 'fun': lambda x, sec_p_min_norm=sec_p_min_norm: x[19] - sec_p_min_norm},  # sec_p > sec_p_min_norm
            {'type': 'ineq', 'fun': lambda x, pry_v_min_norm=pry_v_min_norm: x[20] - pry_v_min_norm},  # pry_v > pry_v_min_norm
            {'type': 'ineq', 'fun': lambda x, sec_v_min_norm=sec_v_min_norm: x[21] - sec_v_min_norm},  # sec_v > sec_v_min_norm
            {'type': 'ineq', 'fun': lambda x, total_coal_min_norm=total_coal_min_norm: x[22] - total_coal_min_norm},  # total_coal > total_coal_min_norm
            {'type': 'ineq', 'fun': lambda x, a_f_ratio_min_norm=a_f_ratio_min_norm: x[23] - a_f_ratio_min_norm},  # a_f_ratio > a_f_ratio_min_norm

            # Constraints equality (tidak berubah)
            {'type': 'eq', 'fun': lambda x, kg_tco_exact_norm=kg_tco_exact_norm: kg_tco_exact_norm - x[24]},  # kg_tco tidak berubah

            # Constraints input >= min_norm
            {'type': 'ineq', 'fun': lambda x, reductor_ratio_min_norm=reductor_ratio_min_norm: x[25] - reductor_ratio_min_norm},  # reductor_ratio > reductor_ratio_min_norm
            {'type': 'ineq', 'fun': lambda x, reductor_consume_min_norm=reductor_consume_min_norm: x[26] - reductor_consume_min_norm},  # reductor_consume > reductor_consume_min_norm

            # Constraints equality (tidak berubah)
            {'type': 'eq', 'fun': lambda x, charge_kiln_exact_norm=charge_kiln_exact_norm: charge_kiln_exact_norm - x[27]},  # charge_kiln tidak berubah

            # Constraints input >= min_norm dan <= max_norm
            {'type': 'ineq', 'fun': lambda x, t_tic162_min_norm=t_tic162_min_norm: x[28] - t_tic162_min_norm},  # t_tic162 > t_tic162_min_norm
            {'type': 'ineq', 'fun': lambda x, t_tic162_max_norm=t_tic162_max_norm: t_tic162_max_norm - x[28]},  # t_tic162 < t_tic162_max_norm
            {'type': 'ineq', 'fun': lambda x, t_tic163_min_norm=t_tic163_min_norm: x[29] - t_tic163_min_norm},  # t_tic163 > t_tic163_min_norm
            {'type': 'ineq', 'fun': lambda x, t_tic163_max_norm=t_tic163_max_norm: t_tic163_max_norm - x[29]},  # t_tic163 < t_tic163_max_norm
            {'type': 'ineq', 'fun': lambda x, t_tic166_min_norm=t_tic166_min_norm: x[30] - t_tic166_min_norm},  # t_tic166 > t_tic166_min_norm
            {'type': 'ineq', 'fun': lambda x, t_tic166_max_norm=t_tic166_max_norm: t_tic166_max_norm - x[30]},  # t_tic166 < t_tic166_max_norm
            {'type': 'ineq', 'fun': lambda x, t_tic172_min_norm=t_tic172_min_norm: x[31] - t_tic172_min_norm},  # t_tic172 > t_tic172_min_norm
            {'type': 'ineq', 'fun': lambda x, t_tic172_max_norm=t_tic172_max_norm: t_tic172_max_norm - x[31]},  # t_tic172 < t_tic172_max_norm

            # Constraints berdasarkan prediksi model Ridge
            {'type': 'ineq', 'fun': lambda x, T_furnace_norm=T_furnace_norm: ridge_model.predict(x.reshape(1, -1))[0][0] - T_furnace_norm},  # T_furnace > T_furnace_norm
            {'type': 'ineq', 'fun': lambda x, ni_met_norm=ni_met_norm: ridge_model.predict(x.reshape(1, -1))[0][1] - ni_met_norm},    # Ni_met > ni_met_norm
            {'type': 'ineq', 'fun': lambda x, C_met_low_norm=C_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][2] - C_met_low_norm},  # C_met > C_met_low_norm
            {'type': 'ineq', 'fun': lambda x, Si_met_low_norm=Si_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][3] - Si_met_low_norm},  # Si_met > Si_met_low_norm
            {'type': 'ineq', 'fun': lambda x, Si_met_high_norm=Si_met_high_norm: Si_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][3]},  # Si_met < Si_met_high_norm
            {'type': 'ineq', 'fun': lambda x, fe_met_low_norm=fe_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][4] - fe_met_low_norm},  # Fe_met > fe_met_low_norm
            {'type': 'ineq', 'fun': lambda x, s_met_low_norm=s_met_low_norm: ridge_model.predict(x.reshape(1, -1))[0][5] - s_met_low_norm},  # S_met > s_met_low_norm
            {'type': 'ineq', 'fun': lambda x, s_met_high_norm=s_met_high_norm: s_met_high_norm - ridge_model.predict(x.reshape(1, -1))[0][5]},  # S_met < s_met_high_norm
            {'type': 'ineq', 'fun': lambda x, ni_slag_low_norm=ni_slag_low_norm: ridge_model.predict(x.reshape(1, -1))[0][6] - ni_slag_low_norm},  # Ni_slag > ni_slag_low_norm
            {'type': 'ineq', 'fun': lambda x, fe_slag_low_norm=fe_slag_low_norm: ridge_model.predict(x.reshape(1, -1))[0][7] - fe_slag_low_norm},  # Fe_slag > fe_slag_low_norm
            {'type': 'ineq', 'fun': lambda x, T_kalsin_low_norm=T_kalsin_low_norm: ridge_model.predict(x.reshape(1, -1))[0][8] - T_kalsin_low_norm},    # T_kalsin > T_kalsin_low_norm
            {'type': 'ineq', 'fun': lambda x, pic_161_low_norm=pic_161_low_norm: ridge_model.predict(x.reshape(1, -1))[0][9] - pic_161_low_norm},  # pic_161 > pic_161_low_norm
            {'type': 'ineq', 'fun': lambda x, loi_kalsin_high=loi_kalsin_high: loi_kalsin_high - ridge_model.predict(x.reshape(1, -1))[0][10]},  # loi_kalsin < loi_kalsin_high
            {'type': 'ineq', 'fun': lambda x, loi_kalsin_low=loi_kalsin_low: ridge_model.predict(x.reshape(1, -1))[0][10] - loi_kalsin_low },  # loi_kalsin > loi_kalsin_low
        ]

        # Melakukan optimisasi menggunakan metode SLSQP dengan constraints yang didefinisikan
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            constraints=constraints_sample,
            options={'ftol': 1e-4}
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
    print(optimized_x_df_original["voltage"])
    print(optimized_x_df_original["current"])
    print(optimized_x_df_original["load"])
    print(optimized_x_df_original["rpm"])
    print(optimized_x_df_original["pry_p"])
    print(optimized_x_df_original["sec_p"])
    print(optimized_x_df_original["pry_v"])
    print(optimized_x_df_original["sec_v"])
    print(optimized_x_df_original["total_coal"])
    print(optimized_x_df_original["a_f_ratio"])
    print(optimized_x_df_original["kg_tco"])
    print(optimized_x_df_original["reductor_ratio"])
    print(optimized_x_df_original["reductor_consume"])
    print(optimized_x_df_original["t_tic162"])
    print(optimized_x_df_original["t_tic163"])

    # Mengambil nilai dari DataFrame dan menyusunnya dalam dictionary dengan nama variabel yang lebih jelas
    output_y = {
        "Temperatur Metal (Temp Metal)": round(optimized_y_df_original["furnace_temp"].values[0], 3),
        "Kandungan Ni di Metal (Ni Met)": round(optimized_y_df_original["ni_met"].values[0], 3),
        "Kandungan C di Metal (C Met)": round(optimized_y_df_original["c_met"].values[0], 3),
        "Kandungan Si di Metal (Si Met)": round(optimized_y_df_original["si_met"].values[0], 3),
        "Kandungan Fe di Metal (Fe Met)": round(optimized_y_df_original["fe_met"].values[0], 3),
        "Kandungan S di Metal (S Met)": round(optimized_y_df_original["s_met"].values[0], 3),
        "Kandungan Ni di Slag (Ni Slag)": round(optimized_y_df_original["ni_slag"].values[0], 3),
        "Kandungan Fe di Slag (Fe Slag)": round(optimized_y_df_original["fe_slag"].values[0], 3),
        "Temperatur Kalsinasi (T Kalsin)": round(optimized_y_df_original["t_kalsin"].values[0], 3),
        "Temperatur PIC 161": round(optimized_y_df_original["pic_161"].values[0], 3),
        "Loss on Ignition Kalsinasi (LOI Kalsin)": round(optimized_y_df_original["loi_kalsin"].values[0], 3)
    }

    output = {
        "Tegangan (Volt)": round(optimized_x_df_original["voltage"].values[0], 3),
        "Arus (Kilo Ampere)": round(optimized_x_df_original["current"].values[0], 3),
        "Beban Listrik (Load MW)": round(optimized_x_df_original["load"].values[0], 3),
        "Kecepatan Putar (RPM)": round(optimized_x_df_original["rpm"].values[0], 3),
        "Aliran Udara Katup Primer (Pry_p)": round(optimized_x_df_original["pry_p"].values[0], 3),
        "Aliran Udara Katup Sekunder (Sec_p)": round(optimized_x_df_original["sec_p"].values[0], 3),
        "Tekanan Udara Primer (Pry_v)": round(optimized_x_df_original["pry_v"].values[0], 3),
        "Tekanan Udara Sekunder (Sec_v)": round(optimized_x_df_original["sec_v"].values[0], 3),
        "Total Konsumsi Batu Bara (Total Coal)": round(optimized_x_df_original["total_coal"].values[0], 3),
        "Rasio Udara-Bahan Bakar (A/F Ratio)": round(optimized_x_df_original["a_f_ratio"].values[0], 3),
        "Rasio Reduktor": round(optimized_x_df_original["reductor_ratio"].values[0], 3),
        "Konsumsi Reduktor (Reductor Consume)": round(optimized_x_df_original["reductor_consume"].values[0], 3),
        "Temperatur TIC 162": round(optimized_x_df_original["t_tic162"].values[0], 3),
        "Temperatur TIC 163": round(optimized_x_df_original["t_tic163"].values[0], 3),
        # "Temperatur TIC 166": round(optimized_x_df_original["t_tic166"].values[0], 3),
        # "Temperatur TIC 172": round(optimized_x_df_original["t_tic172"].values[0], 3)
    }

    # Menyusun output menjadi tabel HTML untuk output pertama (output)
    output_html_1 = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"
    output_html_1 += "<tr><th style='padding: 8px; text-align: left;'>Parameter</th><th style='padding: 8px; text-align: left;'>Value</th></tr>"

    for param, value in output.items():
        formatted_value = f"{value:.2f}".replace('.', ',')  # Format dengan 3 angka di belakang koma, ubah titik menjadi koma
        output_html_1 += f"<tr><td style='padding: 8px; border: 1px solid black;'>{param}</td><td style='padding: 8px; border: 1px solid black;'>{formatted_value}</td></tr>"

    output_html_1 += "</table>"

    # Menyusun output menjadi tabel HTML untuk output kedua (output_y)
    output_html_2 = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"
    output_html_2 += "<tr><th style='padding: 8px; text-align: left;'>Parameter</th><th style='padding: 8px; text-align: left;'>Value</th></tr>"

    for param, value in output_y.items():
        formatted_value = f"{value:.2f}".replace('.', ',')  # Format dengan 3 angka di belakang koma, ubah titik menjadi koma
        output_html_2 += f"<tr><td style='padding: 8px; border: 1px solid black;'>{param}</td><td style='padding: 8px; border: 1px solid black;'>{formatted_value}</td></tr>"

    output_html_2 += "</table>"

    # Menampilkan kedua tabel HTML di Streamlit
    st.markdown("### Input", unsafe_allow_html=True)
    st.markdown(output_html_1, unsafe_allow_html=True)

    st.markdown("### Output Target", unsafe_allow_html=True)
    st.markdown(output_html_2, unsafe_allow_html=True)
