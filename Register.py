#!/usr/bin/python3

import csv
import tkinter as tk
import pandas as pd
# from gpiozero import LED
from mtcnn import MTCNN
from time import sleep
import os


top = tk.Tk() 
top.geometry("600x450+343+144")
top.configure(background="#282828")
# light = LED(17)
 
Name_Entry , ID_Entry = " " , " "
df = pd.read_csv("IFB-FaceReq-Attendance/Database.csv")
name_list = list(df["NAME"])
def clear_window(window):
    for ele in window.winfo_children():
        ele.destroy()

n = 1 

        
def image_capture():
    global Name_Entry
    global ID_Entry
    import cv2
    # light.on()

    check = False

    name = Name_Entry.get()
    ID = str(ID_Entry.get())

    if name == "" or ID == "":
        print("Please Enter valid details")
        check = True
    if name in name_list:
        print("Name already exists")
        check = True
    if check == False :

        os.mkdir("IFB-FaceReq-Attendance/Faces/"+name)

        cam = cv2.VideoCapture(0)

        count = 1
        detector = MTCNN()


        while True:

            if  count == 21:
                break
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break

            result = detector.detect_faces(frame)

            if result != []:
                for person in result:
                    bounding_box = person['box']
                
                cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)

                frame1 = frame[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
        
                img_name = "IFB-FaceReq-Attendance/Faces/{}/{}.png".format(name,ID+"-"+str(count))
                try:
                    cv2.imwrite(img_name, frame1)
                    cv2.imshow(img_name, frame1)
                    print("{} written!".format(img_name))
                except:
                    print("Could not capture that specific frame trying again")

                count += 1
            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break


        cam.release()
        
        cv2.destroyAllWindows()
        # light.off()
        
        Write(name,ID)
    else:
        Register()

def Write(name,ID):
    with open("IFB-FaceReq-Attendance/Database.csv","a") as f:
        write = csv.writer(f)
        write.writerow([name,ID])
    Register() 
def Register():
    #Register
    global n
    global Name_Entry
    global ID_Entry
    
    
    
    top.title("Register")
##    Back_Button = tk.Button(top)
##    Back_Button.place(relx=0.083, rely=0.6, height=31, width=80)
##    Back_Button.configure(activebackground="#32CD32")
##    Back_Button.configure(background="#0269A4")
##    Back_Button.configure(highlightbackground="#0269A4")
##    Back_Button.configure(borderwidth="0")
##    Back_Button.configure(relief="flat")
##    Back_Button.configure(text='''Back''')
##    Back_Button.configure(command = back)

    Capture_Button = tk.Button(top)
    Capture_Button.place(relx=0.783, rely=0.6, height=31, width=80)
    Capture_Button.configure(activebackground="#32CD32")
    Capture_Button.configure(background="#0269A4")
    Capture_Button.configure(highlightbackground="#0269A4")
    Capture_Button.configure(borderwidth="0")
    Capture_Button.configure(relief="flat")
    Capture_Button.configure(text='''Capture''')
    n = 1
    Capture_Button.configure(command = image_capture)

    Name_Label = tk.Label(top)
    Name_Label.place(relx=0.383, rely=0.111, height=31, width=119)
    Name_Label.configure(background="#282828")
    Name_Label.configure(font="Pacifico")
    Name_Label.configure(foreground="#ffffff")
    Name_Label.configure(text='''Name''')

    Name_Entry = tk.Entry(top)
    Name_Entry.place(relx=0.35, rely=0.2,height=23, relwidth=0.3)
    Name_Entry.configure(borderwidth="0")
    Name_Entry.configure(highlightbackground="#ffffff")
    Name_Entry.configure(highlightbackground="#ffffff")

    ID_Label = tk.Label(top)
    ID_Label.place(relx=0.383, rely=0.311, height=31, width=119)
    ID_Label.configure(activebackground="#f9f9f9")
    ID_Label.configure(background="#282828")
    ID_Label.configure(cursor="fleur")
    ID_Label.configure(font="Pacifico")
    ID_Label.configure(foreground="#ffffff")
    ID_Label.configure(text='''Employee ID''')

    ID_Entry = tk.Entry(top)
    ID_Entry.place(relx=0.35, rely=0.4,height=23, relwidth=0.3)
    ID_Entry.configure(background="white")
    ID_Entry.configure(borderwidth="0")
    ID_Entry.configure(font="TkFixedFont")
    ID_Entry.configure(highlightbackground="#ffffff")
    ID_Entry.configure(selectbackground="#c4c4c4")
    top.mainloop()

Register()















######while True:
######    print()
######    print("""Choose an option""")
######    print("""1)Register employee
######2)Exit""")
######
######    n = int(input(" :> "))
######
######    if n == 1:
######        name = input("Please enter name : ")
######        ID = input("Please enter id : ")
######        print("Please press SPACE to capture image")
######        print()
######        image_capture(name)
######        with open("Database.csv","a") as f:
######            write = csv.writer(f)
######            write.writerow([name,ID])
######    elif n == 2:
######        break
######    else:
######        print("Invalid option entered")


  
