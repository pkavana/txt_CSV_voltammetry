# Voltammetry Data Processor (Kavanagh Lab, QUB)

A handy GUI-based tool developed by the **Kavanagh Lab at Queen’s University Belfast** for quick and flexible processing of voltammetric data.

Whether you're working with **cyclic voltammetry**, **linear sweep**, or **rotating disk experiments**, this application allows you to:
- Import `.txt` or `.csv` files using custom delimiters (tab, comma, semicolon, space)
- Plot data using either:
  - Common voltage ranges (shared x-axis)
  - Individual voltage ranges (independent x-axes)
- Export results to Excel:
  - Single worksheet (with voltage/current pairs for each scan)
  - Multiple worksheets (one per scan)
- View clean, automatic plots of your data using `matplotlib`

The tool is built using `Tkinter`, `Pandas`, and `Matplotlib`. No coding knowledge required.

---

## 🔧 Features

- ✅ GUI — no terminal or scripts needed
- ✅ Customizable import (delimiter + file format)
- ✅ Export to Excel with flexible structure
- ✅ Designed for electrochemists and students
- ✅ Ready for bundling into a `.exe` with a custom icon

---

## 🚀 Getting Started

### 🖥 Option 1: Run from Python

1. Clone or download this repository
2. Open a terminal in the folder and run:
   ```bash
   pip install pandas matplotlib
   python volt_processor_gui.py
