import tkinter as tk
import numpy as np


def hyperparameters_selection(name, var, n):

    def on_option_selected_1(*args):
        selected_option_1.set(selected_var_1.get())

    def on_option_selected_2(*args):
        selected_option_2.set(selected_var_2.get())

    def on_option_selected_3(*args):
        selected_option_3.set(selected_var_3.get())

    def on_option_selected_4(*args):
        selected_option_4.set(selected_var_4.get())

    def on_option_selected_5(*args):
        selected_option_5.set(selected_var_5.get())

    def on_option_selected_6(*args):
        selected_option_6.set(selected_var_6.get())

    def on_option_selected_7(*args):
        selected_option_7.set(selected_var_7.get())

    def on_option_selected_8(*args):
        selected_option_8.set(selected_var_8.get())

    def on_option_selected_9(selected_option_9):
        global epochs
        epochs = selected_option_9.get()

    def default_values(name):
        if name == "poisson":
            preset = [3, 2, 2, 2, 1, 1, 3, 2]
        elif name == "advection":
            preset = [3, 2, 2, 2, 3, 1, 3, 1]
        elif name == "graetz":
            preset = [3, 2, 2, 2, 1, 3, 3, 1]
        elif name == "navier_stokes":
            preset = [3, 2, 0, 2, 3, 3, 2, 2]
        elif name == "diffusion":
            preset = [3, 2, 2, 2, 1, 1, 3, 2]
        elif name == "poiseuille":
            preset = [3, 2, 2, 2, 1, 1, 3, 2]
        elif name == "elasticity":
            preset = [3, 2, 2, 2, 1, 1, 3, 2]
        elif name == "stokes_u":
            preset = [3, 2, 2, 2, 1, 3, 3, 1]
        return preset


    preset = default_values(name)
    preset_options_1 = ["sample", "feature", "feature-sampling", "sampling-feature"] 
    preset_options_2 = ["minmax", "robust", "standard"]
    preset_options_3 = ["10", "20", "30", "40", "50"]
    preset_options_4 = ["50", "100", "200", "300", "400"]
    preset_options_5 = ["25", "50", "75", "100", "125"]
    preset_options_6 = ["10", "15", "20", "25", "30"]
    preset_options_7 = ["0.01", "0.1", "1", "10", "100"]
    preset_options_8 = ["1", "2", "3", "4", "5"]
    preset_options_9 = ["5000"]

    try:
        root=tk.Tk()
        root.title("Hyperparameter selection for " + name + " problem")
        root.eval('tk::PlaceWindow . center')

        #BOX1
        row = 0
        labelText_1=tk.StringVar()
        labelText_1.set("Scaling type")
        labelBox_1=tk.Label(root, textvariable=labelText_1)
        labelBox_1.grid(row=row, column=0, columnspan=2)
        selected_var_1 = tk.StringVar(root)  # Variable to store the selected option
        selected_var_1.set(preset_options_1[preset[0]])  # Set the default value
        selected_option_1 = tk.StringVar(root)
        selected_option_1.set(selected_var_1.get())
        dropdown_menu_1 = tk.OptionMenu(root, selected_var_1, *preset_options_1)  # Create and place the OptionMenu widget
        dropdown_menu_1.grid(row=row, column=5, columnspan=3)
        selected_var_1.trace("w", on_option_selected_1)  # Bind the on_option_selected function to the OptionMenu widget

        # BOX2
        row = 1
        labelText_2=tk.StringVar()
        labelText_2.set("Scaling function")
        labelBox_2=tk.Label(root, textvariable=labelText_2)
        labelBox_2.grid(row=row, column=0, columnspan=2)
        selected_var_2 = tk.StringVar(root)  # Variable to store the selected option
        selected_var_2.set(preset_options_2[preset[1]])  # Set the default value
        selected_option_2 = tk.StringVar(root)
        selected_option_2.set(selected_var_2.get())
        dropdown_menu_2 = tk.OptionMenu(root, selected_var_2, *preset_options_2)  # Create and place the OptionMenu widget
        dropdown_menu_2.grid(row=row, column=5, columnspan=3)
        selected_var_2.trace("w", on_option_selected_2)  # Bind the on_option_selected function to the OptionMenu widget

        # BOLEAN BUTTON
        row = 2
        flag1 = tk.BooleanVar()
        flag1.set(True) #set check state
        chk1 = tk.Checkbutton(root, text='Skip connection', var=flag1)
        chk1.grid(row=row, column=5, columnspan=3)

        # RADIO BUTTON
        row = 3
        labelText_3=tk.StringVar()
        labelText_3.set("Training rates")
        labelBox_3=tk.Label(root, textvariable=labelText_3)
        labelBox_3.grid(row=row, column=0, columnspan=2)
        selected_var_3 = tk.IntVar(root)
        selected_var_3.set(preset_options_3[preset[2]])
        selected_option_3 = tk.IntVar()
        selected_option_3.set(selected_var_3.get())
        radio_button1_3 = tk.Radiobutton(root, text="10", variable=selected_var_3, value=10, command=on_option_selected_3)
        radio_button1_3.grid(row=row, column=2, columnspan=2)
        radio_button2_3 = tk.Radiobutton(root, text="20", variable=selected_var_3, value=20, command=on_option_selected_3)
        radio_button2_3.grid(row=row, column=4, columnspan=2)
        radio_button3_3 = tk.Radiobutton(root, text="30", variable=selected_var_3, value=30, command=on_option_selected_3)
        radio_button3_3.grid(row=row, column=6, columnspan=2)
        radio_button4_3 = tk.Radiobutton(root, text="40", variable=selected_var_3, value=40, command=on_option_selected_3)
        radio_button4_3.grid(row=row, column=8, columnspan=2)
        radio_button5_3 = tk.Radiobutton(root, text="50", variable=selected_var_3, value=50, command=on_option_selected_3)
        radio_button5_3.grid(row=row, column=10, columnspan=2)

        # RADIO BUTTON
        row = 4
        labelText_4=tk.StringVar()
        labelText_4.set("Feedforward nodes")
        labelBox_4=tk.Label(root, textvariable=labelText_4)
        labelBox_4.grid(row=row, column=0, columnspan=2)
        selected_var_4 = tk.IntVar(root)
        selected_var_4.set(preset_options_4[preset[3]])
        selected_option_4 = tk.IntVar()
        selected_option_4.set(selected_var_4.get())
        radio_button1_4 = tk.Radiobutton(root, text="50", variable=selected_var_4, value=50, command=on_option_selected_4)
        radio_button1_4.grid(row=row, column=2, columnspan=2)
        radio_button2_4 = tk.Radiobutton(root, text="100", variable=selected_var_4, value=100, command=on_option_selected_4)
        radio_button2_4.grid(row=row, column=4, columnspan=2)
        radio_button3_4 = tk.Radiobutton(root, text="200", variable=selected_var_4, value=200, command=on_option_selected_4)
        radio_button3_4.grid(row=row, column=6, columnspan=2)
        radio_button4_4 = tk.Radiobutton(root, text="300", variable=selected_var_4, value=300, command=on_option_selected_4)
        radio_button4_4.grid(row=row, column=8, columnspan=2)
        radio_button5_4 = tk.Radiobutton(root, text="400", variable=selected_var_4, value=400, command=on_option_selected_4)
        radio_button5_4.grid(row=row, column=10, columnspan=2)

        # RADIO BUTTON
        row = 5
        labelText_5=tk.StringVar()
        labelText_5.set("Map nodes")
        labelBox_5=tk.Label(root, textvariable=labelText_5)
        labelBox_5.grid(row=row, column=0, columnspan=2)
        selected_var_5 = tk.IntVar(root)
        selected_var_5.set(preset_options_5[preset[4]])
        selected_option_5 = tk.IntVar()
        selected_option_5.set(selected_var_5.get())
        radio_button1_5 = tk.Radiobutton(root, text="25", variable=selected_var_5, value=25, command=on_option_selected_5)
        radio_button1_5.grid(row=row, column=2, columnspan=2)
        radio_button2_5 = tk.Radiobutton(root, text="50", variable=selected_var_5, value=50, command=on_option_selected_5)
        radio_button2_5.grid(row=row, column=4, columnspan=2)
        radio_button3_5 = tk.Radiobutton(root, text="75", variable=selected_var_5, value=75, command=on_option_selected_5)
        radio_button3_5.grid(row=row, column=6, columnspan=2)
        radio_button4_5 = tk.Radiobutton(root, text="100", variable=selected_var_5, value=100, command=on_option_selected_5)
        radio_button4_5.grid(row=row, column=8, columnspan=2)
        radio_button5_5 = tk.Radiobutton(root, text="125", variable=selected_var_5, value=125, command=on_option_selected_5)
        radio_button5_5.grid(row=row, column=10, columnspan=2)

        # RADIO BUTTON
        row = 6
        labelText_6=tk.StringVar()
        labelText_6.set("Bottleneck nodes")
        labelBox_6=tk.Label(root, textvariable=labelText_6)
        labelBox_6.grid(row=row, column=0, columnspan=2)
        selected_var_6 = tk.IntVar(root)
        selected_var_6.set(preset_options_6[preset[5]])
        selected_option_6 = tk.IntVar()
        selected_option_6.set(selected_var_6.get())
        radio_button1_6 = tk.Radiobutton(root, text="10", variable=selected_var_6, value=10, command=on_option_selected_6)
        radio_button1_6.grid(row=row, column=2, columnspan=2)
        radio_button2_6 = tk.Radiobutton(root, text="15", variable=selected_var_6, value=15, command=on_option_selected_6)
        radio_button2_6.grid(row=row, column=4, columnspan=2)
        radio_button3_6 = tk.Radiobutton(root, text="20", variable=selected_var_6, value=20, command=on_option_selected_6)
        radio_button3_6.grid(row=row, column=6, columnspan=2)
        radio_button4_6 = tk.Radiobutton(root, text="25", variable=selected_var_6, value=25, command=on_option_selected_6)
        radio_button4_6.grid(row=row, column=8, columnspan=2)
        radio_button5_6 = tk.Radiobutton(root, text="30", variable=selected_var_6, value=30, command=on_option_selected_6)
        radio_button5_6.grid(row=row, column=10, columnspan=2)

        # RADIO BUTTON
        row = 7
        labelText_7=tk.StringVar()
        labelText_7.set("Lambda values")
        labelBox_7=tk.Label(root, textvariable=labelText_7)
        labelBox_7.grid(row=row, column=0, columnspan=2)
        selected_var_7 = tk.DoubleVar(root)
        selected_var_7.set(preset_options_7[preset[6]])
        selected_option_7 = tk.DoubleVar()
        selected_option_7.set(selected_var_7.get())    
        radio_button1_7 = tk.Radiobutton(root, text="0.01", variable=selected_var_7, value=0.01, command=on_option_selected_7)
        radio_button1_7.grid(row=row, column=2, columnspan=2)
        radio_button2_7 = tk.Radiobutton(root, text="0.1", variable=selected_var_7, value=0.1, command=on_option_selected_7)
        radio_button2_7.grid(row=row, column=4, columnspan=2)
        radio_button3_7 = tk.Radiobutton(root, text="1", variable=selected_var_7, value=1, command=on_option_selected_7)
        radio_button3_7.grid(row=row, column=6, columnspan=2)
        radio_button4_7 = tk.Radiobutton(root, text="10", variable=selected_var_7, value=10, command=on_option_selected_7)
        radio_button4_7.grid(row=row, column=8, columnspan=2)
        radio_button5_7 = tk.Radiobutton(root, text="100", variable=selected_var_7, value=100, command=on_option_selected_7)
        radio_button5_7.grid(row=row, column=10, columnspan=2)

        # RADIO BUTTON
        row = 8
        labelText_8=tk.StringVar()
        labelText_8.set("Hidden channels")
        labelBox_8=tk.Label(root, textvariable=labelText_8)
        labelBox_8.grid(row=row, column=0, columnspan=2)
        selected_var_8 = tk.IntVar(root)
        selected_var_8.set(preset_options_8[preset[7]])
        selected_option_8 = tk.IntVar()
        selected_option_8.set(selected_var_8.get())
        radio_button1_8 = tk.Radiobutton(root, text="1", variable=selected_var_8, value=1, command=on_option_selected_8)
        radio_button1_8.grid(row=row, column=2, columnspan=2)
        radio_button2_8 = tk.Radiobutton(root, text="2", variable=selected_var_8, value=2, command=on_option_selected_8)
        radio_button2_8.grid(row=row, column=4, columnspan=2)
        radio_button3_8 = tk.Radiobutton(root, text="3", variable=selected_var_8, value=3, command=on_option_selected_8)
        radio_button3_8.grid(row=row, column=6, columnspan=2)
        radio_button4_8 = tk.Radiobutton(root, text="4", variable=selected_var_8, value=4, command=on_option_selected_8)
        radio_button4_8.grid(row=row, column=8, columnspan=2)
        radio_button5_8 = tk.Radiobutton(root, text="5", variable=selected_var_8, value=5, command=on_option_selected_8)
        radio_button5_8.grid(row=row, column=10, columnspan=2)

        # INSERT BUTTON
        row = 9
        labelText_9=tk.StringVar()
        labelText_9.set("Epochs")
        labelBox_9=tk.Label(root, textvariable=labelText_9)
        labelBox_9.grid(row=row, column=0, columnspan=2)
        selected_option_9 = tk.StringVar()
        selected_option_9.trace("w", lambda name, index, mode, selected_option_9=selected_option_9: on_option_selected_9(selected_option_9))
        entry = tk.Entry(root, textvariable=selected_option_9, width=7)
        entry.insert(0, str(preset_options_9[0])) 
        entry.grid(row=row, column=5, columnspan=3)

        # Fix the width of the first column
        root.columnconfigure(0, minsize=200, weight=1)
        for i in range(12):
            root.columnconfigure(i, minsize=50, weight=1)

        # MAIN LOOP
        root.mainloop()

        selected_ind_1 = preset_options_1.index(selected_var_1.get()) + 1
        selected_ind_2 = preset_options_2.index(selected_var_2.get()) + 1
        flag1 = int(flag1.get() == True)
        argv = [name, var, selected_ind_1, selected_ind_2, flag1,
                selected_option_3.get(), selected_option_4.get(), selected_option_5.get(),
                selected_option_6.get(), selected_option_7.get(), selected_option_8.get(), n, int(selected_option_9.get())]

    except tk.TclError:
        argv = [name, var, preset[0] + 1, preset[1] + 1, 1,
                int(preset_options_3[preset[2]]), int(preset_options_4[preset[3]]), int(preset_options_5[preset[4]]),
                int(preset_options_6[preset[5]]), float(preset_options_7[preset[6]]), int(preset_options_8[preset[7]]), n, int(preset_options_9[0])]

    return argv