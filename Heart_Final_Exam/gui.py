import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import numpy as np

# Load model
model = pickle.load(open("mymodel.sav", "rb"))


def predict():
    try:
        values = [
            float(entry_age.get()),
            float(sex_box.get()[0]),
            float(cp_box.get()),
            float(entry_trestbps.get()),
            float(entry_chol.get()),
            float(fbs_box.get()),
            float(restecg_box.get()),
            float(entry_thalach.get()),
            float(exang_box.get()),
            float(entry_oldpeak.get()),
            float(slope_box.get()),
            float(ca_box.get()),
            float(thal_box.get())
        ]
        result = model.predict([values])[0]
        msg = "Patient has heart disease." if result == 1 else "Patient is healthy."
        messagebox.showinfo("Prediction", msg)
    except Exception as e:
        messagebox.showerror("Error", f"Please enter valid inputs.\n\n{e}")


# Main window
root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("700x800")
root.configure(bg="#d2ede8")

# Font
LABEL_FONT = ("Courier New", 10)

# TTK Button Style
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton",
                font=("Segoe UI", 11, "bold"),
                foreground="white",
                background="#197278",
                borderwidth=0)
style.map("TButton", background=[("active", "#005f73")])

# Frame
form_frame = tk.Frame(root, bg="#d2ede8")
form_frame.place(relx=0.5, rely=0.05, anchor="n")


def make_entry(row, label_with_hint):
    tk.Label(form_frame, text=label_with_hint, font=LABEL_FONT, bg="#d2ede8", fg="#023047").grid(row=row, column=0,
                                                                                                 sticky="w", padx=20,
                                                                                                 pady=6)
    entry = ttk.Entry(form_frame, width=24)
    entry.grid(row=row, column=1, padx=10, pady=6)
    return entry


def make_combo(row, label_with_hint, values):
    tk.Label(form_frame, text=label_with_hint, font=LABEL_FONT, bg="#d2ede8", fg="#023047").grid(row=row, column=0,
                                                                                                 sticky="w", padx=20,
                                                                                                 pady=6)
    box = ttk.Combobox(form_frame, values=values, state="readonly", width=22)
    box.grid(row=row, column=1, padx=10, pady=6)
    box.current(0)
    return box


# Inputs with aligned labels and inline hints
entry_age = make_entry(0, "Age            (29–77)")
sex_box = make_combo(1, "Sex            (0=F, 1=M)", ["0=Female", "1=Male"])
cp_box = make_combo(2, "Chest Pain     (0–3)", ["0", "1", "2", "3"])
entry_trestbps = make_entry(3, "Rest BP        (94–200)")
entry_chol = make_entry(4, "Cholesterol    (126–564)")
fbs_box = make_combo(5, "FBS > 120      (0 or 1)", ["0", "1"])
restecg_box = make_combo(6, "Rest ECG       (0–2)", ["0", "1", "2"])
entry_thalach = make_entry(7, "Max HR         (71–202)")
exang_box = make_combo(8, "Exercise Angina(0 or 1)", ["0", "1"])
entry_oldpeak = make_entry(9, "Oldpeak        (0.0–6.2)")
slope_box = make_combo(10, "Slope          (0–2)", ["0", "1", "2"])
ca_box = make_combo(11, "Vessels (ca)   (0–3)", ["0", "1", "2", "3"])
thal_box = make_combo(12, "Thal           (0–3)", ["0", "1", "2", "3"])

# Predict button
predict_button = ttk.Button(form_frame, text="Predict", command=predict, style="TButton")
predict_button.grid(row=13, column=0, columnspan=2, pady=30)

root.mainloop()
