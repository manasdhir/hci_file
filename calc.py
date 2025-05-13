import tkinter as tk        
import math                  

def append_entry(entry, char):
    entry.insert(tk.END, char)

def evaluate_entry(entry):
    try:
        result = str(eval(entry.get()))
        entry.delete(0, tk.END)
        entry.insert(tk.END, result)
    except Exception:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")

def backspace(entry):
    end_pos = entry.index(tk.END)
    if end_pos > 0:
        entry.delete(end_pos - 1, end_pos)

def clear(entry):
    entry.delete(0, tk.END)

def basic_calculator():
    win = tk.Toplevel(root)
    win.title("Basic Calculator")
    entry = tk.Entry(win, font=("Arial", 18), bd=2, justify='right')
    entry.grid(row=0, column=0, columnspan=4, padx=5, pady=5)
    buttons = [
        '7','8','9','/', '4','5','6','*', '1','2','3','-', '0','.','=','+', '⌫','C'
    ]
    for i, char in enumerate(buttons):
        row = 1 + i//4; col = i%4
        if char == '⌫': cmd = lambda e=entry: backspace(e)
        elif char == 'C': cmd = lambda e=entry: clear(e)
        elif char == '=': cmd = lambda e=entry: evaluate_entry(e)
        else: cmd = lambda ch=char, e=entry: append_entry(e, ch)
        tk.Button(win, text=char, width=5, height=2, command=cmd).grid(row=row, column=col, padx=2, pady=2)

def scientific_calculator():
    win = tk.Toplevel(root)
    win.title("Scientific Calculator")
    entry = tk.Entry(win, font=("Helvetica", 16), bd=2, justify='right')
    entry.grid(row=0, column=0, columnspan=6, padx=5, pady=5)
    keys = ['7','8','9','/','(',')', '4','5','6','*','^','', '1','2','3','-','.','', '0','C','=','+','⌫','']
    for i, char in enumerate(keys):
        if not char: continue
        row = 1 + i//6; col = i%6
        if char == '⌫': cmd = lambda e=entry: backspace(e)
        elif char == 'C': cmd = lambda e=entry: clear(e)
        elif char == '=': cmd = lambda e=entry: evaluate_entry(e)
        elif char == '^': cmd = lambda e=entry: append_entry(e, '**')
        else: cmd = lambda ch=char, e=entry: append_entry(e, ch)
        tk.Button(win, text=char, width=4, height=2, command=cmd).grid(row=row, column=col, padx=1, pady=1)
    funcs = [('sin', math.sin), ('cos', math.cos), ('tan', math.tan),
             ('log', math.log10), ('ln', math.log), ('exp', math.exp),
             ('sqrt', math.sqrt), ('^2', lambda x: x**2)]
    for i, (lbl, fn) in enumerate(funcs):
        r = 1 + i//4; c = 6 + i%4
        tk.Button(win, text=lbl, width=6, height=2,
                  command=lambda f=fn, e=entry: _apply_scientific(e, f)).grid(row=r, column=c, padx=2, pady=2)

def _apply_scientific(entry, func):
    try:
        val = float(entry.get())
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(func(val)))
    except Exception:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")


def user_10_20():
    win = tk.Toplevel(root)
    win.title("Teen Calculator 😃")
    entry = tk.Entry(win, font=("Comic Sans MS", 20), fg="blue", justify='right')
    entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
    buttons = ['7','8','9','+','4','5','6','-','1','2','3','*','0','.','=','/','⌫','C']
    for i, char in enumerate(buttons):
        row = 1 + i//4; col = i%4
        if char == '⌫': cmd = lambda e=entry: backspace(e)
        elif char == 'C': cmd = lambda e=entry: clear(e)
        elif char == '=': cmd = lambda e=entry: evaluate_entry(e)
        else: cmd = lambda ch=char, e=entry: append_entry(e, ch)
        tk.Button(win, text=char, font=("Comic Sans MS", 14), width=4, height=2,
                  bg='#ccf2ff' if char not in ['=', 'C'] else '#ffcc00',
                  command=cmd).grid(row=row, column=col, padx=2, pady=2)


def user_20_plus():
    win = tk.Toplevel(root)
    win.title("Pro Calculator")
    win.configure(bg="#f0f0f0")
    win.geometry("450x500")
    
    display_frame = tk.Frame(win, bg="#f0f0f0")
    display_frame.pack(fill=tk.X, padx=10, pady=10)
    
    buttons_frame = tk.Frame(win, bg="#f0f0f0")
    buttons_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    history_display = tk.Label(display_frame, font=("Segoe UI", 12), anchor='e', bg="#f0f0f0", fg="#666666")
    history_display.pack(fill=tk.X)
    
    entry = tk.Entry(display_frame, font=("Segoe UI", 24), bd=1, justify='right', relief=tk.FLAT,
                    bg="white", fg="#333333")
    entry.pack(fill=tk.X, ipady=10)
    
    for i in range(6):
        buttons_frame.columnconfigure(i, weight=1)
    for i in range(6):
        buttons_frame.rowconfigure(i, weight=1)
    
    num_btn_style = {"bg": "white", "fg": "#333333", "font": ("Segoe UI", 14), 
                     "relief": tk.RAISED, "bd": 1, "padx": 5, "pady": 5, "width": 4, "height": 2}
    op_btn_style = {"bg": "#e6e6e6", "fg": "#333333", "font": ("Segoe UI", 14),
                   "relief": tk.RAISED, "bd": 1, "padx": 5, "pady": 5, "width": 4, "height": 2}
    func_btn_style = {"bg": "#d9edf7", "fg": "#31708f", "font": ("Segoe UI", 12),
                     "relief": tk.RAISED, "bd": 1, "padx": 5, "pady": 5, "width": 4, "height": 2}
    special_btn_style = {"bg": "#dff0d8", "fg": "#3c763d", "font": ("Segoe UI", 14),
                        "relief": tk.RAISED, "bd": 1, "padx": 5, "pady": 5, "width": 4, "height": 2}
    clear_btn_style = {"bg": "#f2dede", "fg": "#a94442", "font": ("Segoe UI", 14),
                      "relief": tk.RAISED, "bd": 1, "padx": 5, "pady": 5, "width": 4, "height": 2}
    
    buttons = [
        ("⌫", 0, 0, clear_btn_style), ("C", 1, 0, clear_btn_style),
        ("(", 2, 0, op_btn_style), (")", 3, 0, op_btn_style),
        ("%", 4, 0, op_btn_style), ("÷", 5, 0, op_btn_style),
        
        ("7", 0, 1, num_btn_style), ("8", 1, 1, num_btn_style), 
        ("9", 2, 1, num_btn_style), ("×", 3, 1, op_btn_style),
        ("x²", 4, 1, func_btn_style), ("√", 5, 1, func_btn_style),
        
        ("4", 0, 2, num_btn_style), ("5", 1, 2, num_btn_style), 
        ("6", 2, 2, num_btn_style), ("-", 3, 2, op_btn_style),
        ("sin", 4, 2, func_btn_style), ("cos", 5, 2, func_btn_style),
        
        ("1", 0, 3, num_btn_style), ("2", 1, 3, num_btn_style), 
        ("3", 2, 3, num_btn_style), ("+", 3, 3, op_btn_style),
        ("tan", 4, 3, func_btn_style), ("log", 5, 3, func_btn_style),
        
        ("0", 0, 4, num_btn_style), (".", 1, 4, num_btn_style), 
        ("±", 2, 4, op_btn_style), ("=", 3, 4, special_btn_style),
        ("π", 4, 4, func_btn_style), ("e", 5, 4, func_btn_style),
    ]
    
    for (char, col, row, style) in buttons:
        if char == "⌫":
            cmd = lambda e=entry: backspace(e)
        elif char == "C":
            cmd = lambda e=entry, h=history_display: clear_all(e, h)
        elif char == "=":
            cmd = lambda e=entry, h=history_display: calculate_with_history(e, h)
        elif char == "×":
            cmd = lambda e=entry: append_entry(e, "*")
        elif char == "÷":
            cmd = lambda e=entry: append_entry(e, "/")
        elif char == "x²":
            cmd = lambda e=entry: square_function(e)
        elif char == "√":
            cmd = lambda e=entry: sqrt_function(e)
        elif char == "sin":
            cmd = lambda e=entry: trig_function(e, math.sin)
        elif char == "cos":
            cmd = lambda e=entry: trig_function(e, math.cos)
        elif char == "tan":
            cmd = lambda e=entry: trig_function(e, math.tan)
        elif char == "log":
            cmd = lambda e=entry: trig_function(e, math.log10)
        elif char == "π":
            cmd = lambda e=entry: append_entry(e, str(math.pi))
        elif char == "e":
            cmd = lambda e=entry: append_entry(e, str(math.e))
        elif char == "x^y":
            cmd = lambda e=entry: append_entry(e, "**")
        elif char == "1/x":
            cmd = lambda e=entry: reciprocal_function(e)
        elif char == "±":
            cmd = lambda e=entry: negate_function(e)
        elif char == "%":
            cmd = lambda e=entry: percent_function(e)
        else:
            cmd = lambda ch=char, e=entry: append_entry(e, ch)
            
        btn = tk.Button(buttons_frame, text=char, command=cmd, **style)
        btn.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")

def clear_all(entry, history):
    entry.delete(0, tk.END)
    history.config(text="")
    
def calculate_with_history(entry, history):
    expression = entry.get()
    history.config(text=expression)
    evaluate_entry(entry)
    
def square_function(entry):
    try:
        value = float(entry.get())
        result = value ** 2
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(result))
    except:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")
        
def sqrt_function(entry):
    try:
        value = float(entry.get())
        result = math.sqrt(value)
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(result))
    except:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")
        
def trig_function(entry, func):
    try:
        value = float(entry.get())
        result = func(value)
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(result))
    except:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")
        
def reciprocal_function(entry):
    try:
        value = float(entry.get())
        result = 1 / value
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(result))
    except:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")
        
def negate_function(entry):
    try:
        value = float(entry.get())
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(-value))
    except:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")
        
def percent_function(entry):
    try:
        value = float(entry.get())
        result = value / 100
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(result))
    except:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")

root = tk.Tk()
root.title("Calculator Launcher")
for i,(label,func) in enumerate([
    ("Basic", basic_calculator),
    ("Scientific", scientific_calculator),
    ("Ages 10-20", user_10_20),
    ("Ages 20+", user_20_plus)
]):
    tk.Button(root, text=label, width=20, command=func).grid(row=i, column=0, pady=5)
root.mainloop()