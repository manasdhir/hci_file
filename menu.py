import tkinter as tk
from tkinter import messagebox

# Create the main application window
root = tk.Tk()
root.title("Menu Types Example")
root.geometry("400x300")

# ----------- a. Menu Bar and b. Pull-Down Menu -----------

# Create a menu bar
menu_bar = tk.Menu(root)

# Create a pull-down menu named 'Actions Menu' with unique labels
actions_menu = tk.Menu(menu_bar, tearoff=0)
actions_menu.add_command(label="Create New Project", command=lambda: messagebox.showinfo("Action", "New Project Created"))
actions_menu.add_command(label="Load Existing Project", command=lambda: messagebox.showinfo("Action", "Project Loaded"))
actions_menu.add_separator()
actions_menu.add_command(label="Close App", command=root.quit)

# Add the 'Actions Menu' to the menu bar
menu_bar.add_cascade(label="Actions Menu", menu=actions_menu)

# ----------- c. Cascading Menu -----------

# Create a 'Features Menu'
features_menu = tk.Menu(menu_bar, tearoff=0)

# Create a cascading submenu named 'Advanced Tools'
advanced_tools = tk.Menu(features_menu, tearoff=0)
advanced_tools.add_command(label="Run Diagnostics", command=lambda: messagebox.showinfo("Advanced Tools", "Diagnostics Started"))
advanced_tools.add_command(label="Generate Report", command=lambda: messagebox.showinfo("Advanced Tools", "Report Generated"))

# Add cascading submenu to 'Features Menu'
features_menu.add_cascade(label="Advanced Tools", menu=advanced_tools)

# Add 'Features Menu' to menu bar
menu_bar.add_cascade(label="Features Menu", menu=features_menu)

# Set the menu bar in the window
root.config(menu=menu_bar)

# ----------- d. Pop-up Menu -----------

# Create a pop-up menu with unique labels
popup_menu = tk.Menu(root, tearoff=0)
popup_menu.add_command(label="Quick Save", command=lambda: messagebox.showinfo("Pop-up", "Saved Quickly"))
popup_menu.add_command(label="Quick Share", command=lambda: messagebox.showinfo("Pop-up", "Shared Successfully"))
popup_menu.add_command(label="Quick Exit", command=root.quit)

# Function to show the pop-up menu
def show_popup(event):
    popup_menu.post(event.x_root, event.y_root)

# Bind right-click to show pop-up menu
root.bind("<Button-3>", show_popup)

# Start the Tkinter event loop
root.mainloop()
