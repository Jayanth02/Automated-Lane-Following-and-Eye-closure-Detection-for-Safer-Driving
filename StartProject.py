import sys
import os.path
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk
from tkinter import filedialog
try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True
import Home  # Assuming Home module is imported for Home.vp_start_gui()


def start_gui():
    '''Starting point when module is the main routine.'''
    global val, main_window, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = tk.Tk()
    main_window = Toplevel1(root)
    root.mainloop()


main_window = None


def create_main_window(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global main_window, window, rt
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    rt = root
    window = tk.Toplevel(root)
    main_window = Toplevel1(window)
    return window, main_window


def destroy_main_window():
    global window
    window.destroy()
    window = None


class Toplevel1:
    def __init__(self, top=None):

        def btn_exit(event):
            top.destroy()
            sys.exit()

        def login_check():
            user_name = self.txt_username.get()
            password = self.txt_password.get()
            if user_name == '' or password == '':
                self.lbl_credential_chk.configure(text='Please Provide all details')
            else:
                if user_name.lower() == 'admin' and password.lower() == 'pass123':
                    return 'Access'
                else:
                    self.lbl_credential_chk.configure(text='Incorrect Username & Password')
                    return 'Denied'

        def btn_compression(event):
            result = login_check()
            if result == 'Access':
                top.destroy()
                Home.start_gui()

        '''This class configures and populates the top-level window.
           top is the top-level containing window.'''
        top.geometry("905x620")
        top.title("New Toplevel")
        top.configure(background="#d9d9d9")

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=0.0, rely=-0.016, height=630, width=905)
        self.Label1.configure(background="#d9d9d9")
        photo_location = os.path.join(prog_location, r"images\\back - Copy.png")
        self._img0 = tk.PhotoImage(file=photo_location)
        self.Label1.configure(image=self._img0)
        self.Label1.configure(text='''Label''')

        self.txt_username = tk.Entry(top)
        self.txt_username.place(relx=0.125, rely=0.350, height=44, relwidth=0.281)
        self.txt_username.configure(background="white")
        self.txt_username.configure(font="-family {Courier New} -size 13 -weight normal -slant roman -underline 0 -overstrike 0")
        self.txt_username.configure(justify='center')

        self.txt_password = tk.Entry(top)
        self.txt_password.place(relx=0.125, rely=0.478, height=44, relwidth=0.281)
        self.txt_password.configure(background="white")
        self.txt_password.configure(font="-family {Courier New} -size 13 -weight normal -slant roman -underline 0 -overstrike 0")
        self.txt_password.configure(justify='center')
        self.txt_password.configure(show='#')

        self.lbl_credential_chk = tk.Label(top)
        self.lbl_credential_chk.place(relx=0.120, rely=0.580, height=34, width=260)
        self.lbl_credential_chk.configure(background="white")
        self.lbl_credential_chk.configure(font="-family {Segoe UI} -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.lbl_credential_chk.configure(foreground="#ff0f3f")
        self.lbl_credential_chk.configure(text='''Credentials''')

        self.btn_comparison = tk.Button(top)
        self.btn_comparison.place(relx=0.22, rely=0.650, height=86, width=86)
        self.btn_comparison.configure(activebackground="#ececec")
        self.btn_comparison.configure(background="#ffffff")
        photo_location = os.path.join(prog_location, r"images\\Algorithm-comparisson_icon.png")
        self._img2 = tk.PhotoImage(file=photo_location)
        self.btn_comparison.configure(image=self._img2)
        self.btn_comparison.configure(relief="flat")
        self.btn_comparison.bind('<Button-1>', btn_compression)

        self.lbl_credential = tk.Label(top)
        self.lbl_credential.place(relx=0.22, rely=0.790, height=34, width=86)
        self.lbl_credential.configure(background="white")
        self.lbl_credential.configure(font="-family {Segoe UI} -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.lbl_credential.configure(foreground="blue")
        self.lbl_credential.configure(text='''Start System''')

        self.btn_log_out = tk.Button(top)
        self.btn_log_out.place(relx=0.1580, rely=0.860, height=35, width=200)
        self.btn_log_out.configure(activebackground="#ececec")
        self.btn_log_out.configure(background="blue")
        self.btn_log_out.configure(font="-family {Segoe UI} -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.btn_log_out.configure(foreground="#ffffff")
        self.btn_log_out.configure(text='''Log Out''')
        self.btn_log_out.bind('<Button-1>', btn_exit)


start_gui()
