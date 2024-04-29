import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from Algorithm import detect_image, detect_video
from startDrowsy import detectDrowsiness as drowsinessDetector
from yolo import YOLO
Yolo_Detector = YOLO()

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True
import mainGUI_support
import os.path


def start_gui():
    '''Starting point when module is the main routine.'''
    global root
    '''Starting point when module is the main routine.'''
    global val, w, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = tk.Tk()
    top = TopLevel1(root)
    mainGUI_support.init(root, top)
    root.mainloop()


class TopLevel1:
    def __init__(self, top=None):

        def image_module_gui(event):
            current_project_folder = os.path.dirname(os.path.abspath(__file__))
            image_file_name = filedialog.askopenfilename(
                initialdir=current_project_folder,
                title="Select Image file",
                filetypes=(("all files", "*.*"), ("PNG files", "*.png"), ("JPG files", "*.jpg"), ("JPEG files", "*.jpeg"))
            )
            detect_image(image_file_name, Yolo_Detector)

        def video_module(event):
            current_project_folder = os.path.dirname(os.path.abspath(__file__))
            video_file_name = filedialog.askopenfilename(
                initialdir=current_project_folder,
                title="Select Video file",
                filetypes=(("MP4 files", "*.mp4"), ("avi files", "*.avi"), ("all files", "*.*"))
            )
            print("\n\nPlease wait, loading models..." + video_file_name)
            Thread(target=drowsinessDetector).start()
            detect_video(video_file_name, Yolo_Detector)

        def btn_exit(event):
            import os
            os._exit(0)

        _bgcolor = '#d9d9d9'
        _fgcolor = '#000000'
        _compcolor = '#d9d9d9'
        _ana1color = '#d9d9d9'
        _ana2color = '#ececec'
        font16 = "-family Constantia -size 40 -weight bold -slant " \
                 "roman -underline 0 -overstrike 0"
        font18 = "-family {Sitka Small} -size 15 -weight bold -slant " \
                 "roman -underline 0 -overstrike 0"

        w = 1000
        h = 650
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        top.geometry('%dx%d+%d+%d' % (w, h, x, y))
        top.title("Driver Drowsiness Detection")
        top.configure(background="#ffffff")

        self.label1 = tk.Label(top)
        self.label1.place(relx=0.3, rely=0.01, height=250, width=350)
        self.label1.configure(background="#ffffff")
        self.label1.configure(disabledforeground="#a3a3a3")
        self.label1.configure(foreground="#000000")
        photo_location = os.path.join(prog_location, "Images/yologo_2.png")
        self._img0 = tk.PhotoImage(file=photo_location)
        self.label1.configure(image=self._img0)
        self.label1.configure(text='''Label''')

        self.label2 = tk.Label(top)
        self.label2.place(relx=0.0, rely=0.4, height=88, width=1000)
        self.label2.configure(background="#ffffff")
        self.label2.configure(disabledforeground="#a3a3a3")
        self.label2.configure(font=font16)
        self.label2.configure(foreground="#2365e8")
        self.label2.configure(text='''Driver Drowsiness Detection''')
        self.label2.configure(width=659)

        self.frame1 = tk.Frame(top)
        self.frame1.place(relx=0.03, rely=0.535, relheight=0.402, relwidth=0.94)
        self.frame1.configure(relief='groove')
        self.frame1.configure(borderwidth="7")
        self.frame1.configure(relief="groove")
        self.frame1.configure(background="#ffffff")
        self.frame1.configure(width=955)

        self.btn_image = tk.Label(self.frame1)
        self.btn_image.place(relx=0.410, rely=0.110, height=176, width=172)
        self.btn_image.configure(activebackground="#f9f9f9")
        self.btn_image.configure(activeforeground="black")
        self.btn_image.configure(background="#ffffff")
        self.btn_image.configure(disabledforeground="#a3a3a3")
        self.btn_image.configure(foreground="#000000")
        self.btn_image.configure(highlightbackground="#d9d9d9")
        self.btn_image.configure(highlightcolor="black")
        photo_location = os.path.join(prog_location, "Images/images icon.png")
        self._img2 = tk.PhotoImage(file=photo_location)
        self.btn_image.configure(image=self._img2)
        self.btn_image.configure(text='''Label''')
        self.btn_image.configure(width=172)
        self.btn_image.bind('<Button-1>', image_module_gui)

        self.btn_video = tk.Label(self.frame1)
        self.btn_video.place(relx=0.042, rely=0.090, height=186, width=162)
        self.btn_video.configure(activebackground="#f9f9f9")
        self.btn_video.configure(activeforeground="black")
        self.btn_video.configure(background="#ffffff")
        self.btn_video.configure(disabledforeground="#a3a3a3")
        self.btn_video.configure(foreground="#000000")
        self.btn_video.configure(highlightbackground="#d9d9d9")
        self.btn_video.configure(highlightcolor="black")
        photo_location = os.path.join(prog_location, "Images/video-camera-png-icon-5.png")
        self._img3 = tk.PhotoImage(file=photo_location)
        self.btn_video.configure(image=self._img3)
        self.btn_video.configure(text='''Label''')
        self.btn_video.configure(width=162)
        self.btn_video.bind('<Button-1>', video_module)

        self.label3_6 = tk.Label(self.frame1)
        self.label3_6.place(relx=0.420, rely=0.784, height=36, width=142)
        self.label3_6.configure(activebackground="#f9f9f9")
        self.label3_6.configure(activeforeground="black")
        self.label3_6.configure(background="#ffffff")
        self.label3_6.configure(disabledforeground="#a3a3a3")
        self.label3_6.configure(font=font18)
        self.label3_6.configure(foreground="#061104")
        self.label3_6.configure(highlightbackground="#d9d9d9")
        self.label3_6.configure(highlightcolor="#000000")
        self.label3_6.configure(text='''Image''')
        self.label3_6.configure(width=142)

        self.label3_6 = tk.Label(self.frame1)
        self.label3_6.place(relx=0.047, rely=0.784, height=36, width=142)
        self.label3_6.configure(activebackground="#f9f9f9")
        self.label3_6.configure(activeforeground="black")
        self.label3_6.configure(background="#ffffff")
        self.label3_6.configure(disabledforeground="#a3a3a3")
        self.label3_6.configure(font=font18)
        self.label3_6.configure(foreground="#061104")
        self.label3_6.configure(highlightbackground="#d9d9d9")
        self.label3_6.configure(highlightcolor="#000000")
        self.label3_6.configure(text='''Video''')
        self.label3_6.configure(width=142)

        self.btn_exit = tk.Label(self.frame1)
        self.btn_exit.place(relx=0.822, rely=0.100, height=186, width=150)
        self.btn_exit.configure(activebackground="#f9f9f9")
        self.btn_exit.configure(activeforeground="black")
        self.btn_exit.configure(background="#ffffff")
        self.btn_exit.configure(disabledforeground="#a3a3a3")
        self.btn_exit.configure(foreground="#000000")
        self.btn_exit.configure(highlightbackground="#d9d9d9")
        self.btn_exit.configure(highlightcolor="black")
        photo_location = os.path.join(prog_location, "Images/ExitIcon.png")
        self._img4 = tk.PhotoImage(file=photo_location)
        self.btn_exit.configure(image=self._img4)
        self.btn_exit.configure(text='''Label''')
        self.btn_exit.configure(width=162)
        self.btn_exit.bind('<Button-1>', btn_exit)

        self.label3_6 = tk.Label(self.frame1)
        self.label3_6.place(relx=0.832, rely=0.784, height=26, width=130)
        self.label3_6.configure(activebackground="#f9f9f9")
        self.label3_6.configure(activeforeground="black")
        self.label3_6.configure(background="#ffffff")
        self.label3_6.configure(disabledforeground="#a3a3a3")
        self.label3_6.configure(font=font18)
        self.label3_6.configure(foreground="#061104")
        self.label3_6.configure(highlightbackground="#d9d9d9")
        self.label3_6.configure(highlightcolor="#000000")
        self.label3_6.configure(text='''Exit''')
        self.label3_6.configure(width=142)


if __name__ == '__main__':
    start_gui()



