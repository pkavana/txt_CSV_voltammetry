import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import re
import numpy as np
import shutil  # For removing directories

def analyze_file(file_path, delimiter="Auto-detect"):
    """Analyze file contents to determine the appropriate loading method"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            preview_lines = [f.readline().strip() for _ in range(10) if f]
        
        sep_dict = {
            "Tab (\\t)": "\t",
            "Comma (,)": ",",
            "Semicolon (;)": ";",
            "Space ( )": " ",
            "Auto-detect": None
        }
        sep = sep_dict.get(delimiter, None)
        
        # Count number of potential header rows
        header_rows = 0
        first_data_row = -1
        
        for i, line in enumerate(preview_lines):
            if not line:  # Skip empty lines
                continue
                
            # Split line using the delimiter or auto-detect
            parts = []
            if sep:
                parts = line.split(sep)
            else:  # Try common delimiters
                for test_sep in [',', '\t', ';', ' ']:
                    if test_sep in line:
                        parts = line.split(test_sep)
                        break
                else:
                    parts = [line]
            
            # Check if this line looks like a header
            is_header = False
            is_numeric = True
            
            for part in parts:
                part = part.strip()
                # Check if it contains typical header keywords
                if any(keyword in part.lower() for keyword in ['potential', 'current', 'voltage', 'amp', 'v)', 'a)', 'we(', 'time']):
                    is_header = True
                
                # Check if it can be converted to a number
                try:
                    float(part)
                except ValueError:
                    if part and part.strip():  # Not empty
                        is_numeric = False
            
            if is_header or (not is_numeric and i < 3):  # Consider as header if within first 3 rows
                header_rows += 1
            elif is_numeric and first_data_row == -1:
                first_data_row = i
                break
        
        return {
            'header_rows': header_rows,
            'first_data_row': first_data_row if first_data_row != -1 else header_rows
        }
    except:
        # If analysis fails, return default values
        return {
            'header_rows': 0,
            'first_data_row': 0
        }

def load_file_with_flexible_format(file_path, delimiter, header_row=None):
    """Load a file with flexible format detection"""
    sep_dict = {
        "Tab (\\t)": "\t",
        "Comma (,)": ",",
        "Semicolon (;)": ";",
        "Space ( )": " ",
        "Auto-detect": None
    }
    sep = sep_dict.get(delimiter, None)
    
    # First, analyze the file to determine structure
    file_analysis = analyze_file(file_path, delimiter)
    detected_header_rows = file_analysis['header_rows']
    first_data_row = file_analysis['first_data_row']
    
    # Try to detect header status and read accordingly
    try:
        # If auto-detection found multiple header rows, use this information
        if detected_header_rows > 0 and header_row is None:
            df = pd.read_csv(file_path, sep=sep, skiprows=detected_header_rows, engine='python')
        # Otherwise, use user-specified header settings
        elif header_row is None:
            # With header (default)
            df = pd.read_csv(file_path, sep=sep, engine='python')
        elif header_row == -1:
            # No header
            df = pd.read_csv(file_path, sep=sep, header=None, engine='python')
        else:
            # Specific header row
            df = pd.read_csv(file_path, sep=sep, header=header_row, engine='python')
            
        # Clean column names if they exist
        if df.columns.dtype != 'int64':
            df.columns = df.columns.str.strip()
            
        return df
    except Exception as e:
        # If failed, try alternative approaches
        try:
            # Try without header
            df = pd.read_csv(file_path, sep=sep, header=None, engine='python')
            return df
        except:
            # Try with different encoding
            try:
                df = pd.read_csv(file_path, sep=sep, encoding='latin1', engine='python')
                if df.columns.dtype != 'int64':
                    df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                # Last resort: try to read line by line
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Skip any potential header rows
                    data_lines = lines[first_data_row:]
                    
                    # Try to determine separator from data lines
                    if sep is None:
                        for test_sep in ['\t', ',', ';', ' ']:
                            if any(test_sep in line for line in data_lines[:10]):
                                sep = test_sep
                                break
                        else:
                            sep = '\t'  # Default to tab if no separator found
                    
                    # Parse data manually
                    data = []
                    for line in data_lines:
                        line = line.strip()
                        if line:
                            try:
                                values = line.split(sep)
                                # Try to convert to numeric
                                num_values = []
                                for val in values:
                                    try:
                                        num_values.append(float(val.strip()))
                                    except:
                                        num_values.append(val.strip())
                                data.append(num_values)
                            except:
                                pass
                    
                    # Create DataFrame
                    if data:
                        df = pd.DataFrame(data)
                        return df
                    else:
                        raise Exception(f"No valid data found in {file_path}")
                        
                except Exception as e:
                    raise Exception(f"Could not read file {file_path.name}: {e}")

def detect_columns(df):
    """Detect potential and current columns based on common naming patterns and data analysis"""
    potential_patterns = ['potential', 'volt', 'voltage', 'e(v)', 'e)', 'v)', 'we(', 'v']
    current_patterns = ['current', 'amp', 'i(a)', 'i)', 'a)', 'i']
    
    pot_col = None
    curr_col = None
    
    # If no columns, return position-based indices
    if df.columns.dtype == 'int64':
        # Check if there are at least 2 columns
        if len(df.columns) >= 2:
            # Perform data distribution analysis to determine which column is potential and which is current
            # Typically, potential values are more regular and sequential compared to current values
            
            # Convert columns to numeric if possible
            numeric_cols = []
            for col in df.columns[:2]:  # Check only first two columns
                try:
                    df[col] = pd.to_numeric(df[col])
                    numeric_cols.append(col)
                except:
                    pass
            
            if len(numeric_cols) >= 2:
                # Check data variance and patterns to identify columns
                # Calculate differences between consecutive values for each column
                col0_diffs = np.diff(df[numeric_cols[0]])
                col1_diffs = np.diff(df[numeric_cols[1]])
                
                # Calculate standard deviation of differences
                col0_std = np.std(col0_diffs)
                col1_std = np.std(col1_diffs)
                
                # Calculate mean of absolute values for each column
                col0_mean = np.mean(np.abs(df[numeric_cols[0]]))
                col1_mean = np.mean(np.abs(df[numeric_cols[1]]))
                
                # If one column has much more regular differences (smaller std dev), 
                # it's likely the potential column
                if col0_std < col1_std * 0.5:
                    return numeric_cols[0], numeric_cols[1]
                elif col1_std < col0_std * 0.5:
                    return numeric_cols[1], numeric_cols[0]
                # If values in one column are much larger than the other,
                # the smaller values are likely the current
                elif col0_mean > col1_mean * 100:
                    return numeric_cols[0], numeric_cols[1]
                elif col1_mean > col0_mean * 100:
                    return numeric_cols[1], numeric_cols[0]
                
                # Default to first=potential, second=current if analysis is inconclusive
                return numeric_cols[0], numeric_cols[1]
            
        return 0, 1
        
    # Look for column names matching patterns
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Check for potential column
        if pot_col is None:
            for pattern in potential_patterns:
                if pattern in col_lower:
                    pot_col = col
                    break
        
        # Check for current column
        if curr_col is None:
            for pattern in current_patterns:
                if pattern in col_lower:
                    curr_col = col
                    break
    
    # If we found both columns, return them
    if pot_col is not None and curr_col is not None:
        return pot_col, curr_col
    
    # If we found only one column, try data analysis to find the other
    if pot_col is not None and curr_col is None:
        # Find another column that's not the potential column
        other_cols = [col for col in df.columns if col != pot_col]
        if other_cols:
            return pot_col, other_cols[0]
    
    if pot_col is None and curr_col is not None:
        # Find another column that's not the current column
        other_cols = [col for col in df.columns if col != curr_col]
        if other_cols:
            return other_cols[0], curr_col
    
    # Fallback: assume first column is potential, second is current
    if len(df.columns) >= 2:
        # Verify if this assumption is correct by analyzing data values
        try:
            # Convert to numeric
            col1_data = pd.to_numeric(df[df.columns[0]])
            col2_data = pd.to_numeric(df[df.columns[1]])
            
            # Check which has smaller absolute values (typically current)
            col1_mean = np.mean(np.abs(col1_data))
            col2_mean = np.mean(np.abs(col2_data))
            
            if col1_mean < col2_mean * 0.01:  # If col1 values are much smaller
                return df.columns[1], df.columns[0]  # Swap the columns
                
        except:
            pass  # If conversion fails, use default order
            
        return df.columns[0], df.columns[1]
    
    # Last resort: return positions
    return 0, 1

def preprocess_files(folder_path, output_folder, delimiter, header_row):
    """Preprocess files to standardized format"""
    output_folder.mkdir(exist_ok=True)
    processed_files = []
    
    for file_path in sorted(folder_path.glob("*")):
        if file_path.is_file() and file_path.suffix.lower() in [".txt", ".csv", ""]:
            try:
                # Load with flexible format
                df = load_file_with_flexible_format(file_path, delimiter, header_row)
                
                # Check for purely numeric data (no text)
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                # If we have headers but data is text, it may be multiple headers
                # Try to convert columns to numeric where possible
                if len(numeric_cols) < 2:
                    # Check if first rows might be additional headers
                    for i in range(min(3, len(df))):
                        try:
                            row = df.iloc[i]
                            # Check if this row contains known header keywords
                            row_text = ' '.join(str(x).lower() for x in row)
                            if any(keyword in row_text for keyword in ['potential', 'current', 'voltage', 'we(', 'v)', 'a)']):
                                # Skip this row by dropping it
                                df = df.drop(i).reset_index(drop=True)
                        except:
                            pass
                    
                    # Try to convert columns to numeric now
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except:
                            pass
                
                # Detect potential and current columns
                pot_col, curr_col = detect_columns(df)
                
                # Ensure the data is numeric
                try:
                    df[pot_col] = pd.to_numeric(df[pot_col])
                    df[curr_col] = pd.to_numeric(df[curr_col])
                except:
                    # If conversion fails, try to analyze the data more carefully
                    try:
                        # Sometimes there are header rows mixed with data
                        # Create a copy and try to identify true data rows
                        temp_df = df.copy()
                        numeric_rows = []
                        
                        for i, row in temp_df.iterrows():
                            try:
                                float(row[pot_col])
                                float(row[curr_col])
                                numeric_rows.append(i)
                            except:
                                pass
                        
                        # Use only the numeric rows
                        if numeric_rows:
                            df = df.iloc[numeric_rows].reset_index(drop=True)
                            df[pot_col] = pd.to_numeric(df[pot_col])
                            df[curr_col] = pd.to_numeric(df[curr_col])
                    except:
                        raise Exception(f"Could not convert data to numeric format in {file_path.name}")
                
                # Additional data validation to ensure pot_col is really potential and curr_col is current
                try:
                    # Check if columns might be reversed using scientific notation
                    pot_col_sci_count = sum(1 for x in df[pot_col].astype(str) if 'e-' in x.lower())
                    curr_col_sci_count = sum(1 for x in df[curr_col].astype(str) if 'e-' in x.lower())
                    
                    # If current column doesn't have scientific notation but potential does, swap them
                    if pot_col_sci_count > len(df) * 0.5 and curr_col_sci_count < len(df) * 0.1:
                        pot_col, curr_col = curr_col, pot_col
                
                    # Additional check: compare absolute mean values
                    pot_mean = np.mean(np.abs(df[pot_col]))
                    curr_mean = np.mean(np.abs(df[curr_col]))
                    
                    # Current values are typically much smaller than potential values
                    if pot_mean < curr_mean * 0.01 and curr_mean > 0.1:
                        pot_col, curr_col = curr_col, pot_col
                except:
                    # If analysis fails, use the original detection
                    pass
                
                # Create standardized dataframe
                std_df = pd.DataFrame({
                    "Potential (V)": df[pot_col],
                    "Current (A)": df[curr_col]
                })
                
                # Save to output folder
                output_file = output_folder / f"std_{file_path.name}"
                std_df.to_csv(output_file, index=False)
                processed_files.append(output_file)
                
            except Exception as e:
                print(f"Skipping file {file_path.name}: {e}")
    
    return processed_files

def load_volt_data(processed_files, use_common_voltage):
    """Load standardized voltage data files"""
    combined_df = pd.DataFrame()
    individual_traces = {}
    common_potential = None

    for file_path in processed_files:
        try:
            df = pd.read_csv(file_path)
            
            # Ensure we have the standard column names
            if "Potential (V)" not in df.columns or "Current (A)" not in df.columns:
                # Try to find columns with similar names
                pot_col, curr_col = detect_columns(df)
                
                # Rename columns to standard names
                df = df.rename(columns={pot_col: "Potential (V)", curr_col: "Current (A)"})
            
            # Get a clean label from the filename
            label = file_path.stem
            if label.startswith('std_'):
                label = label[4:]  # Remove 'std_' prefix
                
            # Remove any file extension if present in the label
            for ext in ['.txt', '.csv']:
                if label.endswith(ext):
                    label = label[:-len(ext)]
            
            if use_common_voltage:
                if common_potential is None:
                    common_potential = df["Potential (V)"]
                    combined_df["Potential (V)"] = common_potential
                combined_df[label] = df["Current (A)"]
            else:
                individual_traces[label] = (df["Potential (V)"], df["Current (A)"])
                
        except Exception as e:
            print(f"Error processing standardized file {file_path.name}: {e}")

    return combined_df, individual_traces

def plot_data(combined_df, individual_traces, use_common_voltage):
    """Plot voltammetry data"""
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
    """Main function to process files"""
    folder = filedialog.askdirectory(title="Select Folder Containing Voltammetry Files")
    if not folder:
        return

    try:
        folder_path = Path(folder)
        temp_folder = folder_path / "temp_standardized"
        
        use_common_voltage = (plot_mode.get() == "Common Voltage Range")
        single_worksheet = (worksheet_mode.get() == "Single Worksheet")
        delimiter = delimiter_choice.get()
        header_option = header_choice.get()
        
        # Map header option to actual header row parameter
        header_map = {
            "Auto-detect": None,
            "No header": -1,
            "Header in first row": 0,
            "Header in second row": 1
        }
        header_row = header_map.get(header_option, None)
        
        # Show progress information
        status_var.set("Processing files...")
        root.update()
        
        # Preprocess files to standardize format
        processed_files = preprocess_files(folder_path, temp_folder, delimiter, header_row)
        
        if not processed_files:
            messagebox.showerror("Error", "No valid files found or processed.")
            status_var.set("Ready")
            return
        
        status_var.set(f"Processing {len(processed_files)} files...")
        root.update()
            
        # Process the standardized files
        combined_df, individual_traces = load_volt_data(processed_files, use_common_voltage)
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

        status_var.set("Creating plot...")
        root.update()
        
        plot_data(combined_df, individual_traces, use_common_voltage)
        
        # Clean up temporary files
        status_var.set("Cleaning up temporary files...")
        root.update()
        try:
            if temp_folder.exists():
                shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"Warning: Could not delete temporary files: {e}")
        
        status_var.set("Ready")
        messagebox.showinfo("Success", f"Excel and plot created:\n{output_excel}")

    except Exception as e:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        status_var.set("Error occurred")
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

# === GUI Setup ===
root = tk.Tk()
root.title("Kavanagh Lab Voltammetry Processor — Queen's University Belfast")
root.geometry("600x520")

# Set icon path explicitly
icon_path = "C:\\Users\\3051642\\Desktop\\Volt Processor v1\\volt_icon.ico"

if os.path.exists(icon_path):
    root.iconbitmap(icon_path)

# Create frame for better organization
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Status variable
status_var = tk.StringVar()
status_var.set("Ready")

# Plotting mode
ttk.Label(main_frame, text="1. Select Plotting Mode").grid(row=0, column=0, sticky=tk.W, pady=5)
plot_mode = ttk.Combobox(main_frame, values=["Common Voltage Range", "Individual Voltage Ranges"], state="readonly")
plot_mode.set("Common Voltage Range")
plot_mode.grid(row=0, column=1, pady=5, sticky=tk.W+tk.E, padx=5)

# Excel output mode
ttk.Label(main_frame, text="2. Select Excel Output Mode").grid(row=1, column=0, sticky=tk.W, pady=5)
worksheet_mode = ttk.Combobox(main_frame, values=["Single Worksheet", "Multiple Worksheets"], state="readonly")
worksheet_mode.set("Single Worksheet")
worksheet_mode.grid(row=1, column=1, pady=5, sticky=tk.W+tk.E, padx=5)

# File delimiter
ttk.Label(main_frame, text="3. Choose File Delimiter").grid(row=2, column=0, sticky=tk.W, pady=5)
delimiter_choice = ttk.Combobox(main_frame, values=["Tab (\\t)", "Comma (,)", "Semicolon (;)", "Space ( )", "Auto-detect"], state="readonly")
delimiter_choice.set("Auto-detect")
delimiter_choice.grid(row=2, column=1, pady=5, sticky=tk.W+tk.E, padx=5)

# Header options
ttk.Label(main_frame, text="4. Header Options").grid(row=3, column=0, sticky=tk.W, pady=5)
header_choice = ttk.Combobox(main_frame, values=["Auto-detect", "No header", "Header in first row", "Header in second row"], state="readonly")
header_choice.set("Auto-detect")
header_choice.grid(row=3, column=1, pady=5, sticky=tk.W+tk.E, padx=5)

# Process button
ttk.Button(main_frame, text="5. Select Folder and Process", command=process_files).grid(row=4, column=0, columnspan=2, pady=15)

# Status bar
status_bar = ttk.Label(main_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
status_bar.grid(row=5, column=0, columnspan=2, sticky=tk.W+tk.E, pady=5)

# Info text
info_text = """
This program will:
1. Convert all input files to a standard format
2. Automatically detect voltage and current columns
3. Process files with or without headers
4. Create Excel output and plots
5. Automatically clean up temporary files
"""
info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT, wraplength=550)
info_label.grid(row=6, column=0, columnspan=2, pady=10, sticky=tk.W)

# Credits
ttk.Label(main_frame, text="Developed by Kavanagh Lab, Queen's University Belfast", foreground="grey").grid(row=7, column=0, columnspan=2, pady=10)

root.mainloop()