import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys

def load_volt_data(folder_path, use_common_voltage, delimiter):
    combined_df = pd.DataFrame()
    individual_traces = {}
    common_potential = None

    sep_dict = {
        "Tab (\\t)": "\t",
        "Comma (,)": ",",
        "Semicolon (;)": ";",
        "Space ( )": " "
    }
    sep = sep_dict.get(delimiter, "\t")

    for file_path in sorted(folder_path.glob("*")):
        if file_path.is_file() and file_path.suffix.lower() in [".txt", ".csv", ""]:
            try:
                df = pd.read_csv(file_path, sep=sep)
                df.columns = df.columns.str.strip()
                pot_col = [col for col in df.columns if "Potential" in col][0]
                current_col = [col for col in df.columns if "Current" in col][0]

                label = file_path.stem
                if use_common_voltage:
                    if common_potential is None:
                        common_potential = df[pot_col]
                        combined_df["Potential (V)"] = common_potential
                    combined_df[label] = df[current_col]
                else:
                    individual_traces[label] = (df[pot_col], df[current_col])
            except Exception as e:
                print(f"Skipping file {file_path.name}: {e}")

    return combined_df, individual_traces

def plot_data(combined_df, individual_traces, use_common_voltage):
    plt.figure(figsize=(10, 6))
    if use_common_voltage:
        for col in combined_df.columns[1:]:
            plt.plot(combined_df["Potential (V)"], combined_df[col], label=col)
        plt.xlabel("Potential (V)")
        plt.ylabel("Current (A)")
        plt.title("Voltammetry Data (Common Voltage Range)")
    else:
        for label, (x, y) in individual_traces.items():
            plt.plot(x, y, label=label)
        plt.xlabel("Potential (V)")
        plt.ylabel("Current (A)")
        plt.title("Voltammetry Data (Individual Voltage Ranges)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_files():
    folder = filedialog.askdirectory(title="Select Folder Containing Voltammetry Files")
    if not folder:
        return

    try:
        folder_path = Path(folder)
        use_common_voltage = (plot_mode.get() == "Common Voltage Range")
        single_worksheet = (worksheet_mode.get() == "Single Worksheet")
        delimiter = delimiter_choice.get()

        combined_df, individual_traces = load_volt_data(folder_path, use_common_voltage, delimiter)
        output_excel = folder_path / "processed_voltammetry_data.xlsx"

        with pd.ExcelWriter(output_excel) as writer:
            if single_worksheet:
                if use_common_voltage:
                    combined_df.to_excel(writer, sheet_name="Combined", index=False)
                else:
                    big_df = pd.DataFrame()
                    for label, (x, y) in individual_traces.items():
                        temp = pd.DataFrame({
                            f"Potential (V) — {label}": x,
                            f"Current (A) — {label}": y
                        })
                        big_df = pd.concat([big_df, temp], axis=1)
                    big_df.to_excel(writer, sheet_name="Combined", index=False)
            else:
                if use_common_voltage:
                    for col in combined_df.columns[1:]:
                        temp_df = pd.DataFrame({
                            "Potential (V)": combined_df["Potential (V)"],
                            "Current (A)": combined_df[col]
                        })
                        temp_df.to_excel(writer, sheet_name=col[:31], index=False)
                else:
                    for label, (x, y) in individual_traces.items():
                        temp_df = pd.DataFrame({"Potential (V)": x, "Current (A)": y})
                        temp_df.to_excel(writer, sheet_name=label[:31], index=False)

        plot_data(combined_df, individual_traces, use_common_voltage)
        messagebox.showinfo("Success", f"Excel and plot created:\n{output_excel}")

    except Exception as e:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

# === GUI Setup ===
root = tk.Tk()
root.title("Kavanagh Lab Voltammetry Processor — Queen’s University Belfast")
root.geometry("600x420")

# Cross-platform icon support (bundled .exe or .py)
if hasattr(sys, '_MEIPASS'):
    icon_path = os.path.join(sys._MEIPASS, "volt_icon.ico")
else:
    icon_path = "volt_icon.ico"

if os.path.exists(icon_path):
    root.iconbitmap(icon_path)

tk.Label(root, text="1. Select Plotting Mode").pack(pady=5)
plot_mode = ttk.Combobox(root, values=["Common Voltage Range", "Individual Voltage Ranges"], state="readonly")
plot_mode.set("Common Voltage Range")
plot_mode.pack(pady=5)

tk.Label(root, text="2. Select Excel Output Mode").pack(pady=5)
worksheet_mode = ttk.Combobox(root, values=["Single Worksheet", "Multiple Worksheets"], state="readonly")
worksheet_mode.set("Single Worksheet")
worksheet_mode.pack(pady=5)

tk.Label(root, text="3. Choose File Delimiter").pack(pady=5)
delimiter_choice = ttk.Combobox(root, values=["Tab (\\t)", "Comma (,)", "Semicolon (;)", "Space ( )"], state="readonly")
delimiter_choice.set("Tab (\\t)")
delimiter_choice.pack(pady=5)

tk.Label(root, text="4. Click to Process Folder").pack(pady=10)
tk.Button(root, text="Select Folder and Process", command=process_files, padx=10, pady=10).pack()

tk.Label(root, text="\nDeveloped by Kavanagh Lab, Queen’s University Belfast", fg="grey").pack(pady=10)

root.mainloop()
