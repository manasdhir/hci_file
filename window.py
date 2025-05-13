import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ----- Primary Window -----
root = tk.Tk()
root.title("Primary Window")
root.geometry("400x300")

# ----- 1. Dialog Box -----
def open_dialog():
    filedialog.askopenfilename(title="Open File")

# ----- 2. Property Sheet (Toplevel with Tabs) -----
def open_property_sheet():
    win = tk.Toplevel(root)
    win.title("Property Sheet")
    notebook = ttk.Notebook(win)
    
    tab1 = tk.Frame(notebook)
    tab2 = tk.Frame(notebook)
    notebook.add(tab1, text='General')
    notebook.add(tab2, text='Advanced')
    
    tk.Label(tab1, text="Property 1:").pack(pady=5)
    tk.Entry(tab1).pack()
    tk.Label(tab2, text="Property 2:").pack(pady=5)
    tk.Entry(tab2).pack()
    
    notebook.pack(padx=10, pady=10)

# ----- 3. Message Box -----
def show_message():
    messagebox.showinfo("Info", "This is a message box.")

# ----- 4. Palette Window -----
def open_palette():
    palette = tk.Toplevel(root)
    palette.title("Palette")
    palette.geometry("150x100")
    palette.attributes('-topmost', True)
    tk.Button(palette, text="Tool 1").pack(pady=5)
    tk.Button(palette, text="Tool 2").pack()

# ----- 5. Pop-up Menu -----
popup = tk.Menu(root, tearoff=0)
popup.add_command(label="Popup!", command=lambda: messagebox.showinfo("Pop-up", "You right-clicked."))

def show_popup(event):
    popup.post(event.x_root, event.y_root)

root.bind("<Button-3>", show_popup)

# ----- Buttons to Trigger Windows -----
tk.Button(root, text="Open Dialog", command=open_dialog).pack(pady=5)
tk.Button(root, text="Open Property Sheet", command=open_property_sheet).pack(pady=5)
tk.Button(root, text="Show Message Box", command=show_message).pack(pady=5)
tk.Button(root, text="Open Palette Window", command=open_palette).pack(pady=5)

# ----- Run the app -----
root.mainloop()
