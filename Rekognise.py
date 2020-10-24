#########!/usr/bin/python3
########
########import face_recognition
########import cv2
########import numpy as np
########import time
########import pandas as pd
########from datetime import datetime
########import tkinter as tk
########import tkinter.ttk as ttk
########import csv
########import smbus
########import sys
########import getopt
########import pigpio
########import dlib
########import math
########from gpiozero import LED
########
########top = ""
########SLabel1 = ""
########Label1_6 = ""
########log_list = []
########log_list_out = []
########       
########df = pd.read_csv("Database.csv")
########light = LED(17)
########
########print(df)
########
########known_face_names = list(df["NAME"])
########known_face_ids = list(df["ID Number"])
########known_face_encodings = []
########
########for i in range(len(known_face_names)):
########    string1 =  "face_recognition.face_encodings(face_recognition.load_image_file('Faces/"+known_face_names[i]+".png'))[0]"
##########    if i != (len(known_face_names)-(len(known_face_names)-2)):
########    exec("known_face_encodings += ["+string1+"]")
########    
########print(len(known_face_encodings))
########
########known_face_names = list(df["NAME"])
########known_face_ids = list(df["ID Number"])
########face_locations = []
########face_encodings = []
########face_names = []
########process_this_frame = True
########
########
########def temp():
########    i2c_bus = smbus.SMBus(1)
########    OMRON_1=0x0a 					# 7 bit I2C address of Omron MEMS Temp Sensor D6T-44L
########    OMRON_BUFFER_LENGTH=5			# Omron data buffer size
########    temperature_data=[0]*OMRON_BUFFER_LENGTH 	# initialize the temperature data list
########
########    # intialize the pigpio library and socket connection to the daemon (pigpiod)
########    pi = pigpio.pi()              # use defaults
########    version = pi.get_pigpio_version()
########    handle = pi.i2c_open(1, 0x0a) # open Omron D6T device at address 0x0a on bus 1
########
########
########
########    # initialize the device based on Omron's appnote 1
########    result=i2c_bus.write_byte(OMRON_1,0x4c);
########    #print 'write result = '+str(result)
########
########    #for x in range(0, len(temperature_data)):
########      #print x
########      # Read all data  tem
########      #temperature_data[x]=i2c_bus.read_byte(OMRON_1)
########    (bytes_read, temperature_data) = pi.i2c_read_device(handle, len(temperature_data))
########
########    # Display data 
########
########    a=(temperature_data[2]+temperature_data[3]*256)/10
########    a = a*9
########    a = a/5
########    a = a+32
########    a = a+15
########    print(a)
########    #print 'done'
########    pi.i2c_close(handle)
########    pi.stop()
########    return a
########    
########def log_length(log):
########    logs=[]
########    length = 0
########    with open("Logs.csv","r") as f:
########        read = csv.reader(f)
########
########        for i in read:
########            logs+=[i]
########        logs=logs[1:]
########    check = False
########    for j,i in enumerate(logs):
########        if log[1]==i[1] and log[0]==i[0]:
########            check = True
########            length = len(logs[j])
########            
########    return length
########
########
########def logging(log):
########    logs=[]
########    
########    with open("Logs.csv","r") as f:
########        read = csv.reader(f)
########
########        for i in read:
########            logs+=[i]
########    log_head = logs[0]
########    logs=logs[1:]
########     
########    day = datetime.now().date().day
########    month = datetime.now().date().month
########    year = datetime.now().date().year
########
########    date = str(day)+"-"+str(month)+"-"+str(year)
########
########    log_index = 0
########
########    check = False
########    print(log[1])
########    for j,i in enumerate(logs):
########        print(log[1] == i[1])
########        if log[1]==i[1] and log[0]==i[0]:
########            check = True
########            log_index = j
########        
########    if check == False:
########        with open("Logs.csv","a") as f:
########            write = csv.writer(f)
########            print("Writing : ",log)
########            write.writerow(log)
########    elif check == True:
########        if len(log) == 6:
########            check_write = False
########            print("Reached")
########            print(logs[log_index])
########            if len(logs[log_index]) == 7:
########                logs[log_index].pop()
########                logs[log_index].pop()
########                logs[log_index].pop()
########            if len(logs[log_index]) == 6:
########                logs[log_index].pop()
########                logs[log_index].pop()
########            if len(logs[log_index]) == 5:
########                logs[log_index].pop()
##########            if len(logs[log_index])>4:
##########                for i in range(0,(len(logs[log_index]) - 4)):
##########                    logs[log_index].pop()
##########                    print(logs[log_index])
########            
########            out_TIME = datetime.now()
########            in_TIME = datetime.strptime(logs[log_index][2],'%Y-%m-%d %H:%M:%S.%f')
########
########            print(type(out_TIME))
########            print(type(in_TIME))
########
########            working_hours = out_TIME - in_TIME
########            working_hours = datetime.strptime(str(working_hours),"%H:%M:%S.%f")
########            print(working_hours.hour,working_hours.minute)
########            try:
########                working_hours = str(working_hours.hour) + ":" + str(working_hours.minute)
########                logs[log_index] += [log[-2]]
########                logs[log_index] += [log[-1]]
########                logs[log_index] += [working_hours]
########                check_write = True
########            except:
########                working_hours = str(working_hours.minute) + ":" + str(working_hours.second)
########                logs[log_index] += [log[-1]]
########                logs[log_index] += [working_hours]
########                check_write = True
########            logs = [log_head] + logs
########            if check_write == True:
########                with open("Logs.csv","w") as f:
########                    write = csv.writer(f)
########                    print("Writing logs : ",logs)
########                    write.writerows(logs)
########        elif len(log) == 4:
########            if len(logs[log_index]) == 6:
########                logs[log_index].pop()
########                logs[log_index].pop()
########            if len(logs[log_index]) == 5:
########                logs[log_index].pop()
########            if len(logs[log_index]) == 4:
########                logs[log_index].pop()
########                        
########            logs[log_index] += [log[-1]]
########            logs = [log_head] + logs
########            with open("Logs.csv","w") as f:
########                write = csv.writer(f)
########                print("Writing logs : ",logs)
########                write.writerows(logs)
########            
########
########    return check
########
########def test_human(video_capture):
########
################    light.on()
########    
########    result = True
########    BLINK_RATIO_THRESHOLD = 5.7
########    #-----Step 5: Getting to know blink ratio
########
########    def midpoint(point1 ,point2):
########        return (point1.x + point2.x)/2,(point1.y + point2.y)/2
########
########    def euclidean_distance(point1 , point2):
########        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
########
########    def get_blink_ratio(eye_points, facial_landmarks):
########        
########        #loading all the required points
########        corner_left  = (facial_landmarks.part(eye_points[0]).x, 
########                    facial_landmarks.part(eye_points[0]).y)
########        corner_right = (facial_landmarks.part(eye_points[3]).x, 
########                    facial_landmarks.part(eye_points[3]).y)
########
########        center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
########                             facial_landmarks.part(eye_points[2]))
########        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
########                             facial_landmarks.part(eye_points[4]))
########
########        #calculating distance
########        horizontal_length = euclidean_distance(corner_left,corner_right)
########        vertical_length = euclidean_distance(center_top,center_bottom)
########
########        ratio = horizontal_length / vertical_length
########
########        return ratio
########
########    #livestream from the webcam 
############    cap = cv2.VideoCapture(0)
########
########    '''in case of a video
##############    cap = cv2.VideoCapture("__path_of_the_video__")'''
########
########    #name of the display window in OpenCV
########    cv2.namedWindow('BlinkDetector')
########
########    #-----Step 3: Face detection with dlib-----
########    detector = dlib.get_frontal_face_detector()
########
########    #-----Step 4: Detecting Eyes using landmarks in dlib-----
########    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
########    #these landmarks are based on the image above 
########    left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
########    right_eye_landmarks = [42, 43, 44, 45, 46, 47]
########
########    prev_ratio = 0
########    blinks = []
########    count = 0
########
########    while count<20:
########        #capturing frame
########        retval, frame = video_capture.read()
########
########        #exit the application if frame not found
########        if not retval:
########            print("Can't receive frame (stream end?). Exiting ...")
########            break 
########
########        #-----Step 2: converting image to grayscale-----
##########        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
########
########        #-----Step 3: Face detection with dlib-----
########        #detecting faces in the frame
########        faces,_,_ = detector.run(image = frame, upsample_num_times = 0, 
########                       adjust_threshold = 0.0)
########
########        #-----Step 4: Detecting Eyes using landmarks in dlib-----
########        blink_ratio = 0
########        for face in faces:
########            landmarks = predictor(frame, face)
########
########            #-----Step 5: Calculating blink ratio for one eye-----
########            left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
########            right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
########            blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2
########
########            blinks += [blink_ratio]
########            count += 1
########                
########        if blink_ratio > BLINK_RATIO_THRESHOLD:
########            #Blink detected! Do Something!
##########                        cv2.putText(frame,"BLINKING",(10,50), cv2.FONT_HERSHEY_SIMPLEX,
##########                                2,(255,255,255),2,cv2.LINE_AA)
########            print("Blinked with the ratio : ",blink_ratio)
########                
########                
########        cv2.imshow('BlinkDetector', frame)
########        key = cv2.waitKey(1)
########        if key == 27:
########            break
########    deviation = np.std(blinks)
########    if deviation < 0.35:
########            result = False
########    else:
########            result = True
########            
########    #releasing the VideoCapture object
##########    cap.release()
########    cv2.destroyAllWindows()
########    light.off()
########    return result,deviation
########
########def rekognise():
########    global top
########    global SLabel1
########    results = []
########    matches = []
########    best_match_index = 0
########    process_this_frame = True
########    fake = False
########    video_capture = cv2.VideoCapture(0)
########
########    
########    count = 0
########    while True:
########        
########        if count==1:
########            light.on()
########        if count == 9:
########            break
########        
########        
########        # Grab a single frame of video
########        ret, frame = video_capture.read()
########
########        # Resize frame of video to 1/4 size for faster face recognition processing
########        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
########
########        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
########        rgb_small_frame = small_frame[:, :, ::-1]
########            
########        # Find all the faces and face encodings in the current frame of video
########        face_locations = face_recognition.face_locations(rgb_small_frame)
########        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
########
########        face_names = []
########        for face_encoding in face_encodings:
########            # See if the face is a match for the known face(s)
########            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
########            name = "Unknown"
########            # Use the known face with the smallest distance to the new face
########            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
########            best_match_index = np.argmin(face_distances)
########            if matches[best_match_index]:
########                name = known_face_names[best_match_index]
########            face_names.append(name)
########        if True in matches:
##########            video_capture.release()
########            results+=[known_face_names[best_match_index]]
########            count += 1
########    check_human = test_human(video_capture)
########    print(check_human)
########    if check_human[0] == True:
########        count+=1
########        print("Real")
########    else:
########        fake = True
########        
########                
########    if fake == False:      
########        unique_list = list(np.unique(results))
########        count_list = []
########        for i in unique_list:
########            count_list+=[[i,results.count(i)]]
########        print(count_list)
########        max = 0
########        max_index = 0
########        for i,j in enumerate(count_list):
########            if j[1]>max:
########                max_index = i
########
########        
########        print(results)
########        time.sleep(2)
########        clear_window(top)
########        result(unique_list[max_index],known_face_ids[known_face_names.index(unique_list[max_index])])
########    else:
########        show_fake()
########
####################        # Display the resulting image      
####################        cv2.imshow('Video', frame)
####################
####################        # Hit 'q' on the keyboard to quit!
####################        if cv2.waitKey(1) & 0xFF == ord('q'):
####################            break
########
########
########
########
########def show_fake():
########    global top
########    clear_window(top)
########
########    SLabel1 = tk.Label(top)
########    SLabel1.place(relx=0.35, rely=0.311, height=161, width=189)
########    SLabel1.configure(background="red")
########    SLabel1.configure(borderwidth="2")
########    SLabel1.configure(relief="raised")
########    SLabel1.configure(text='''FAKE FACE''')
########    SLabel1.configure(font="Pacifico")
########
########    countdown1(3)
########
########
########
########def start(t=1):
########    global SLabel1
########    global SLabel1_1
########    global SLabel1_2
########    global top
########    
########    if t==1:
########        top = tk.Tk()
########    elif t==0:
########        clear_window(top)
########    print("Starting")
########    top.geometry("600x450+383+141")
########    top.minsize(1, 1)
########    top.maxsize(1351, 738)
########    top.resizable(True,True)
########    top.title("EMPLOYEE ATTENDANCE")
########    top.configure(background="#232323")
########
########    SLabel1 = tk.Label(top)
########    SLabel1.place(relx=0.35, rely=0.311, height=161, width=189)
########    SLabel1.configure(background="#6b6b6b")
########    SLabel1.configure(borderwidth="2")
########    SLabel1.configure(relief="raised")
########    SLabel1.configure(text='''Ready''')
########    SLabel1.configure(font=("Pacifico",30))
########
########    SLabel1_1 = tk.Label(top)
########    SLabel1_1.place(relx=0.233, rely=0.044, height=61, width=329)
########    SLabel1_1.configure(activebackground="#f9f9f9")
########    SLabel1_1.configure(background="#6b6b6b")
########    SLabel1_1.configure(borderwidth="2")
########    SLabel1_1.configure(relief="raised")
########    SLabel1_1.configure(text='''IFB''')
########    SLabel1_1.configure(font=("Pacifico",20))
########
##############    SLabel1_2 = tk.Label(top)
##############    SLabel1_2.place(relx=0.3, rely=0.822, height=51, width=259)
##############    SLabel1_2.configure(activebackground="#f9f9f9")
##############    SLabel1_2.configure(background="#6b6b6b")
##############    SLabel1_2.configure(borderwidth="2")
##############    SLabel1_2.configure(relief="raised")
##############    SLabel1_2.configure(text='''------''')
########    countdown(2)
########    top.mainloop()
########    
########
########def result(NAME,ID,TIME=datetime.now()):
########    global top
########    global Label1_6
########    global log_list
########    global log_list_out
########    clear_window(top)
########    
########    top.geometry("600x478+455+131")
########    top.minsize(1, 1)
########    top.maxsize(1351, 738)
########    top.resizable(1, 1)
########    top.title("RECOGNISED")
########    top.configure(background="#232323")
########    top.configure(highlightbackground="#000000")
########    
########
########    Label1 = tk.Label(top)
########    Label1.place(relx=0.083, rely=0.067, height=44, width=109)
########    Label1.configure(activeforeground="#444444")
########    Label1.configure(background="#6b6b6b")
########    Label1.configure(borderwidth="2")
########    Label1.configure(foreground="#ffffff")
########    Label1.configure(relief="raised")
########    Label1.configure(text='''NAME''')
########    Label1.configure(font="Pacifico")
########
########    Label1_1 = tk.Label(top)
########    Label1_1.place(relx=0.4, rely=0.067, height=44, width=109)
########    Label1_1.configure(activebackground="#f9f9f9")
########    Label1_1.configure(activeforeground="#444444")
########    Label1_1.configure(background="#6b6b6b")
########    Label1_1.configure(borderwidth="2")
########    Label1_1.configure(foreground="#ffffff")
########    Label1_1.configure(relief="raised")
########    Label1_1.configure(text='''EMPLOYEE ID''')
########    Label1_1.configure(font=("Pacifico",10))
########    
########    Label1_2 = tk.Label(top)
########    Label1_2.place(relx=0.717, rely=0.067, height=44, width=109)
########    Label1_2.configure(activebackground="#f9f9f9")
########    Label1_2.configure(activeforeground="#444444")
########    Label1_2.configure(background="#6b6b6b")
########    Label1_2.configure(borderwidth="2")
########    Label1_2.configure(foreground="#ffffff")
########    Label1_2.configure(relief="raised")
########    Label1_2.configure(text='''STATUS''')
########    Label1_2.configure(font=("Pacifico",15))
########
########    Label1_3 = tk.Label(top)
########    Label1_3.place(relx=0.033, rely=0.178, height=107, width=169)
########    Label1_3.configure(activebackground="#f9f9f9")
########    Label1_3.configure(activeforeground="#444444")
########    Label1_3.configure(background="#6b6b6b")
########    Label1_3.configure(borderwidth="2")
########    Label1_3.configure(foreground="#ffffff")
########    Label1_3.configure(relief="raised")
########    Label1_3.configure(text=NAME)
########    Label1_3.configure(font="Pacifico")
########
########    ID = str(ID)
########    ID = "0"*(8-len(ID))+ID
########    Label1_4 = tk.Label(top)
########    Label1_4.place(relx=0.35, rely=0.178, height=107, width=169)
########    Label1_4.configure(activebackground="#f9f9f9")
########    Label1_4.configure(activeforeground="#444444")
########    Label1_4.configure(background="#6b6b6b")
########    Label1_4.configure(borderwidth="2")
########    Label1_4.configure(foreground="#ffffff")
########    Label1_4.configure(relief="raised")
########    Label1_4.configure(text=ID)
########    Label1_4.configure(font="Pacifico")
########
########    Label1_5 = tk.Label(top)
########    Label1_5.place(relx=0.667, rely=0.178, height=107, width=169)
########    Label1_5.configure(activebackground="#f9f9f9")
########    Label1_5.configure(activeforeground="#444444")
########    Label1_5.configure(background="#6b6b6b")
########    Label1_5.configure(borderwidth="2")
########    Label1_5.configure(foreground="#ffffff")
########    Label1_5.configure(relief="raised")
########    Label1_5.configure(font="Pacifico")
########
########    Label1_6 = tk.Label(top)
########    Label1_6.place(relx=0.317, rely=0.502, height=161, width=219)
########    Label1_6.configure(activebackground="#f9f9f9")
########    Label1_6.configure(activeforeground="#444444")
########    Label1_6.configure(background="blue")
########    Label1_6.configure(borderwidth="2")
########    Label1_6.configure(foreground="#ffffff")
########    Label1_6.configure(relief="raised")
########    Label1_6.configure(font="Pacifico")
########
########    day = datetime.now().date().day
########    month = datetime.now().date().month
########    year = datetime.now().date().year
########    date = str(day)+"-"+str(month)+"-"+str(year)
########
########    log_list = [NAME,date,TIME]
########    print(log_list)
########    check1 = logging([NAME,date,TIME])
########
########    if check1 == False:
########        Label1_5.configure(text=str(TIME.hour)+":"+str(TIME.minute))
########        Label1_6.configure(text='''Reading Temperature''')
########        print("Going to countdown")
########        countdown_temp(2)
########    else:
########        log_len = log_length([NAME,date])
########        print("Log len = ",log_len)
########        if log_len == 4 or log_len == 6 or log_len == 7:
########            # logging([NAME,date,TIME,0,TIME])
########            log_list_out = [NAME,date,TIME,0,TIME]
########            Label1_5.configure(text="Already logged in")
########            Label1_6.configure(text="""Reading Temperature""")
########            countdown_temp1(3)
########    
########
##########    if temperature > 100:
##########        Label1_6.configure(background="red")
##########        Label1_6.configure(text="NOT OK")
##########    else:
##########        Label1_6.configure(background="green")
##########        Label1_6.configure(text="OK")
########    
########
########
########def clear_window(window):
########    for ele in window.winfo_children():
########        ele.destroy()
########def countdown(count):
########    global top
########        
########    if count > 0:
########        # call countdown again after 1000ms (1s)
########        top.after(500, countdown, count-1)
########    if count == 0:
########        rekognise()
########def countdown1(count1):
########    global top
########    if count1 > 0:
########        top.after(1000, countdown1, count1-1)
########    if count1 == 0:
########        start(0)
########def countdown_temp(count2):
########    global top
########    global Label1_6
########    global log_list
########    if count2 > 0:
########        top.after(800, countdown_temp , count2-1)
########    if count2 == 0:
########        c = 0
########        t = 0
########        while True:
########            if c==5:
########                break
########
########            temperature = temp()
########            if temperature>92 and temperature<110:
########                t+=temperature
########                c+=1
########            time.sleep(1)
########        print(t/c)
########        log_list += [t/c]
########        if (t/c) > 100:
########            Label1_6.configure(background = "red")
########            Label1_6.configure(font=("Pacifico",20))
########            Label1_6.configure(text=str(round(t/c,1)))
########        else:
########            Label1_6.configure(background = "green")
########            Label1_6.configure(font=("Pacifico",20))
########            Label1_6.configure(text=str(round(t/c,1)))
########        logging(log_list)
########        countdown1(4)
########
########def countdown_temp1(count3):
########    global top
########    global Label1_6
########    global log_list_out
########    if count3 > 0:
########        top.after(800, countdown_temp1 , count3-1)
########    if count3 == 0:
########        c = 0
########        t = 0
########        while True:
########            if c==5:
########                break
########
########            temperature = temp()
########            if temperature>92 and temperature<110:
########                t+=temperature
########                c+=1
########            time.sleep(1)
########        print(t/c)
########        log_list_out += [t/c]
########        if (t/c) > 100:
########            Label1_6.configure(background = "red")
########            Label1_6.configure(font=("Pacifico",20))
########            Label1_6.configure(text=str(round(t/c,1)))
########        else:
########            Label1_6.configure(background = "green")
########            Label1_6.configure(font=("Pacifico",20))
########            Label1_6.configure(text=str(round(t/c,1)))
########        print(log_list_out)            
########        logging(log_list_out)
########        countdown1(4)        
##################################################################################################
##################################################################################################
########
########start()
########
######### Release handle to the webcam
##########video_capture.release()
########
########
########



#!/usr/bin/python3

import face_recognition
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime
import tkinter as tk
import tkinter.ttk as ttk
import csv
import smbus
import sys
import getopt
import pigpio
import dlib
import math
import sqlite3
from gpiozero import LED

top = ""
SLabel1 = ""
Label1_6 = ""
log_list = []
log_list_out = []
       
df = pd.read_csv("Database.csv")
light = LED(17)

print(df)

known_face_names = list(df["NAME"])
known_face_ids = list(df["ID Number"])
known_face_encodings = []

for i in range(len(known_face_names)):
    string1 =  "face_recognition.face_encodings(face_recognition.load_image_file('Faces/"+known_face_names[i]+".png'))[0]"
##    if i != (len(known_face_names)-(len(known_face_names)-2)):
    exec("known_face_encodings += ["+string1+"]")
    
print(len(known_face_encodings))

known_face_names = list(df["NAME"])
known_face_ids = list(df["ID Number"])
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


con = sqlite3.connect("Rekognise_Logs.db")
cur = con.cursor()

try:
    con.execute("""create table Logs(Name char(30),ID varchar(30) , Date date,In_time datetime,In_Temperature float,Out_Time datetime,Out_Temperature float,Working_Hours char(10))""")
except:
    print("Table Exists")

def temp():
    i2c_bus = smbus.SMBus(1)
    OMRON_1=0x0a 					# 7 bit I2C address of Omron MEMS Temp Sensor D6T-44L
    OMRON_BUFFER_LENGTH=5			# Omron data buffer size
    temperature_data=[0]*OMRON_BUFFER_LENGTH 	# initialize the temperature data list

    # intialize the pigpio library and socket connection to the daemon (pigpiod)
    pi = pigpio.pi()              # use defaults
    version = pi.get_pigpio_version()
    handle = pi.i2c_open(1, 0x0a) # open Omron D6T device at address 0x0a on bus 1



    # initialize the device based on Omron's appnote 1
    result=i2c_bus.write_byte(OMRON_1,0x4c);
    #print 'write result = '+str(result)

    #for x in range(0, len(temperature_data)):
      #print x
      # Read all data  tem
      #temperature_data[x]=i2c_bus.read_byte(OMRON_1)
    (bytes_read, temperature_data) = pi.i2c_read_device(handle, len(temperature_data))

    # Display data 

    a=(temperature_data[2]+temperature_data[3]*256)/10
    a = a*9
    a = a/5
    a = a+32
    a = a+15
    print(a)
    #print 'done'
    pi.i2c_close(handle)
    pi.stop()
    return a

###############################################################################
def check_present(log):
    Name = log[0]
    Id = log[1]
    date = log[2]

    lis = []

    command = "Select Name,ID,Date from Logs where Date = '%s' and ID = '%s' " % (date,Id)
    res = cur.execute(command)
    for i in res:
        lis += [i]
    if len(lis) == 1:
        return True
    else:
        return False

def write_log(log):
    if len(log) == 5:       #this is for first login of the day

        for i in range(3):
            log += ["Null"]
        
        cur.executemany("insert into Logs Values(?,?,?,?,?,?,?,?)",[log])
        for i in cur.execute("select * from Logs"):
            print(i)
        con.commit()
    elif len(log) == 7:

        out_time = datetime.strptime(str(datetime.now())[11:],"%H:%M:%S.%f")
        
        res = cur.execute("select In_Time from Logs where ID = '%s' and Date = '%s'" % (log[1],log[2]))
        in_time = datetime.strptime(str(res.fetchall()[0][0]),"%H:%M:%S.%f")
        
        working_hours = out_time - in_time

        days, seconds = working_hours.days, working_hours.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        working_hours = str(hours) + ":" + str(minutes) + ":" + str(seconds)

        print(working_hours)
        cur.execute("update Logs set Out_Time = '%s', Out_Temperature = '%s', Working_Hours = '%s' where ID = '%s' and Date = '%s'" % (str(out_time)[11:],log[6],working_hours,log[1],log[2],))
        con.commit()

#####################################################################################
    
def confirmation(name):
    global top
    clear_window(top)
    top.geometry("600x450+688+255")
    top.minsize(148, 1)
    top.maxsize(1924, 1055)
    top.resizable(1,  1)
    top.title("Confirmation")
    top.configure(background="#232323")
    
    Label1 = tk.Label(top)
    Label1.place(relx=0.283, rely=0.178, height=106, width=262)
    Label1.configure(background="#232323")
    Label1.configure(disabledforeground="#a3a3a3")
    Label1.configure(font=("Pacifico",40))
    Label1.configure(foreground="#ffffff")
    Label1.configure(text=name)

    check = [""]
    def returnTrue():
        check[0] = True
        

    def returnFalse():
        check[0] = False
    

    
    Button_No = tk.Button(top)
    Button_No.place(relx=0.25, rely=0.756, height=75, width=75)
    Button_No.configure(activebackground="#ececec")
    Button_No.configure(font=("Pacifico",20))
    Button_No.configure(activeforeground="#000000")
    Button_No.configure(background="red")
    Button_No.configure(disabledforeground="#a3a3a3")
    Button_No.configure(foreground="#000000")
    Button_No.configure(highlightbackground="#d9d9d9")
    Button_No.configure(highlightcolor="black")
    Button_No.configure(pady="0")
    Button_No.configure(text='''No''')
    Button_No.configure(command = returnFalse)
    
    Button_Yes = tk.Button(top)
    Button_Yes.place(relx=0.625, rely=0.756, height=75, width=75)
    Button_Yes.configure(activebackground="#ececec")
    Button_Yes.configure(font=("Pacifico",20))
    Button_Yes.configure(activeforeground="#000000")
    Button_Yes.configure(background="green")
    Button_Yes.configure(disabledforeground="#a3a3a3")
    Button_Yes.configure(foreground="#000000")
    Button_Yes.configure(highlightbackground="#d9d9d9")
    Button_Yes.configure(highlightcolor="black")
    Button_Yes.configure(pady="0")
    Button_Yes.configure(text='''Yes''')
    Button_Yes.configure(command = returnTrue)
    top.mainloop()
    return check[0]

def test_human(video_capture):

########    light.on()
    
    result = True
    BLINK_RATIO_THRESHOLD = 5.7
    #-----Step 5: Getting to know blink ratio

    def midpoint(point1 ,point2):
        return (point1.x + point2.x)/2,(point1.y + point2.y)/2

    def euclidean_distance(point1 , point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_blink_ratio(eye_points, facial_landmarks):
        
        #loading all the required points
        corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y)
        corner_right = (facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y)

        center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                             facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))

        #calculating distance
        horizontal_length = euclidean_distance(corner_left,corner_right)
        vertical_length = euclidean_distance(center_top,center_bottom)

        ratio = horizontal_length / vertical_length

        return ratio

    #livestream from the webcam 
####    cap = cv2.VideoCapture(0)

    '''in case of a video
######    cap = cv2.VideoCapture("__path_of_the_video__")'''

    #name of the display window in OpenCV
    cv2.namedWindow('BlinkDetector')

    #-----Step 3: Face detection with dlib-----
    detector = dlib.get_frontal_face_detector()

    #-----Step 4: Detecting Eyes using landmarks in dlib-----
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #these landmarks are based on the image above 
    left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
    right_eye_landmarks = [42, 43, 44, 45, 46, 47]

    prev_ratio = 0
    blinks = []
    count = 0

    while count<20:
        #capturing frame
        retval, frame = video_capture.read()

        #exit the application if frame not found
        if not retval:
            print("Can't receive frame (stream end?). Exiting ...")
            break 

        #-----Step 2: converting image to grayscale-----
##        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #-----Step 3: Face detection with dlib-----
        #detecting faces in the frame
        faces,_,_ = detector.run(image = frame, upsample_num_times = 0, 
                       adjust_threshold = 0.0)

        #-----Step 4: Detecting Eyes using landmarks in dlib-----
        blink_ratio = 0
        for face in faces:
            landmarks = predictor(frame, face)

            #-----Step 5: Calculating blink ratio for one eye-----
            left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
            right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
            blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2

            blinks += [blink_ratio]
            count += 1
                
        if blink_ratio > BLINK_RATIO_THRESHOLD:
            #Blink detected! Do Something!
##                        cv2.putText(frame,"BLINKING",(10,50), cv2.FONT_HERSHEY_SIMPLEX,
##                                2,(255,255,255),2,cv2.LINE_AA)
            print("Blinked with the ratio : ",blink_ratio)
                
                
        cv2.imshow('BlinkDetector', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    deviation = np.std(blinks)
    if deviation < 0.35:
            result = False
    else:
            result = True
            
    #releasing the VideoCapture object
##    cap.release()
    cv2.destroyAllWindows()
    light.off()
    return result,deviation

def rekognise():
    global top
    global SLabel1
    results = []
    matches = []
    best_match_index = 0
    process_this_frame = True
    fake = False
    video_capture = cv2.VideoCapture(0)

    
    count = 0
    while True:
        time.sleep(1)
        if count==1:
            light.on()
        if count == 13:
            break
        
        
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
            
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
        if True in matches:
##            video_capture.release()
            results+=[known_face_names[best_match_index]]
            count += 1
    check_human = test_human(video_capture)
    print(check_human)
    if check_human[0] == True:
        count+=1
        print("Real")
    else:
        fake = True
        
                
    if fake == False:      
        unique_list = list(np.unique(results))
        count_list = []
        for i in unique_list:
            count_list+=[[i,results.count(i)]]
        print(count_list)
        max1 = 0
        max_index = 0
        for i,j in enumerate(count_list):
            if j[1]>max1:
                max_index = i
                max1 = j[1]

        
        print(results)
        time.sleep(2)
        clear_window(top)
        confirm = confirmation(count_list[max_index][0])
        if confirm == True:
            result(count_list[max_index][0],known_face_ids[known_face_names.index(count_list[max_index][0])])
        else:
            start(0)
    else:
        show_fake()

############        # Display the resulting image      
############        cv2.imshow('Video', frame)
############
############        # Hit 'q' on the keyboard to quit!
############        if cv2.waitKey(1) & 0xFF == ord('q'):
############            break




def show_fake():
    global top
    clear_window(top)

    SLabel1 = tk.Label(top)
    SLabel1.place(relx=0.35, rely=0.311, height=161, width=189)
    SLabel1.configure(background="red")
    SLabel1.configure(borderwidth="2")
    SLabel1.configure(relief="raised")
    SLabel1.configure(text='''FAKE FACE''')
    SLabel1.configure(font="Pacifico")

    countdown1(3)



def start(t=1):
    global SLabel1
    global SLabel1_1
    global SLabel1_2
    global top
    
    if t==1:
        top = tk.Tk()
    elif t==0:
        clear_window(top)
    print("Starting")
    top.geometry("600x450+383+141")
    top.minsize(1, 1)
    top.maxsize(1351, 738)
    top.resizable(True,True)
    top.title("EMPLOYEE ATTENDANCE")
    top.configure(background="#232323")

    SLabel1 = tk.Label(top)
    SLabel1.place(relx=0.35, rely=0.311, height=161, width=189)
    SLabel1.configure(background="#6b6b6b")
    SLabel1.configure(borderwidth="2")
    SLabel1.configure(relief="raised")
    SLabel1.configure(text='''Ready''')
    SLabel1.configure(font=("Pacifico",30))

    SLabel1_1 = tk.Label(top)
    SLabel1_1.place(relx=0.233, rely=0.044, height=61, width=329)
    SLabel1_1.configure(activebackground="#f9f9f9")
    SLabel1_1.configure(background="#6b6b6b")
    SLabel1_1.configure(borderwidth="2")
    SLabel1_1.configure(relief="raised")
    SLabel1_1.configure(text='''IFB''')
    SLabel1_1.configure(font=("Pacifico",20))

######    SLabel1_2 = tk.Label(top)
######    SLabel1_2.place(relx=0.3, rely=0.822, height=51, width=259)
######    SLabel1_2.configure(activebackground="#f9f9f9")
######    SLabel1_2.configure(background="#6b6b6b")
######    SLabel1_2.configure(borderwidth="2")
######    SLabel1_2.configure(relief="raised")
######    SLabel1_2.configure(text='''------''')
    countdown(2)
    top.mainloop()
    

def result(NAME,ID,TIME=datetime.now()):
    global top
    global Label1_6
    global log_list
    global log_list_out
    clear_window(top)
    
    top.geometry("600x478+455+131")
    top.minsize(1, 1)
    top.maxsize(1351, 738)
    top.resizable(1, 1)
    top.title("RECOGNISED")
    top.configure(background="#232323")
    top.configure(highlightbackground="#000000")
    

    Label1 = tk.Label(top)
    Label1.place(relx=0.083, rely=0.067, height=44, width=109)
    Label1.configure(activeforeground="#444444")
    Label1.configure(background="#6b6b6b")
    Label1.configure(borderwidth="2")
    Label1.configure(foreground="#ffffff")
    Label1.configure(relief="raised")
    Label1.configure(text='''NAME''')
    Label1.configure(font="Pacifico")

    Label1_1 = tk.Label(top)
    Label1_1.place(relx=0.4, rely=0.067, height=44, width=109)
    Label1_1.configure(activebackground="#f9f9f9")
    Label1_1.configure(activeforeground="#444444")
    Label1_1.configure(background="#6b6b6b")
    Label1_1.configure(borderwidth="2")
    Label1_1.configure(foreground="#ffffff")
    Label1_1.configure(relief="raised")
    Label1_1.configure(text='''EMPLOYEE ID''')
    Label1_1.configure(font=("Pacifico",10))
    
    Label1_2 = tk.Label(top)
    Label1_2.place(relx=0.717, rely=0.067, height=44, width=109)
    Label1_2.configure(activebackground="#f9f9f9")
    Label1_2.configure(activeforeground="#444444")
    Label1_2.configure(background="#6b6b6b")
    Label1_2.configure(borderwidth="2")
    Label1_2.configure(foreground="#ffffff")
    Label1_2.configure(relief="raised")
    Label1_2.configure(text='''STATUS''')
    Label1_2.configure(font=("Pacifico",15))

    Label1_3 = tk.Label(top)
    Label1_3.place(relx=0.033, rely=0.178, height=107, width=169)
    Label1_3.configure(activebackground="#f9f9f9")
    Label1_3.configure(activeforeground="#444444")
    Label1_3.configure(background="#6b6b6b")
    Label1_3.configure(borderwidth="2")
    Label1_3.configure(foreground="#ffffff")
    Label1_3.configure(relief="raised")
    Label1_3.configure(text=NAME)
    Label1_3.configure(font="Pacifico")

    ID = str(ID)
    ID = "0"*(8-len(ID))+ID
    Label1_4 = tk.Label(top)
    Label1_4.place(relx=0.35, rely=0.178, height=107, width=169)
    Label1_4.configure(activebackground="#f9f9f9")
    Label1_4.configure(activeforeground="#444444")
    Label1_4.configure(background="#6b6b6b")
    Label1_4.configure(borderwidth="2")
    Label1_4.configure(foreground="#ffffff")
    Label1_4.configure(relief="raised")
    Label1_4.configure(text=ID)
    Label1_4.configure(font="Pacifico")

    Label1_5 = tk.Label(top)
    Label1_5.place(relx=0.667, rely=0.178, height=107, width=169)
    Label1_5.configure(activebackground="#f9f9f9")
    Label1_5.configure(activeforeground="#444444")
    Label1_5.configure(background="#6b6b6b")
    Label1_5.configure(borderwidth="2")
    Label1_5.configure(foreground="#ffffff")
    Label1_5.configure(relief="raised")
    Label1_5.configure(font="Pacifico")

    Label1_6 = tk.Label(top)
    Label1_6.place(relx=0.317, rely=0.502, height=161, width=219)
    Label1_6.configure(activebackground="#f9f9f9")
    Label1_6.configure(activeforeground="#444444")
    Label1_6.configure(background="blue")
    Label1_6.configure(borderwidth="2")
    Label1_6.configure(foreground="#ffffff")
    Label1_6.configure(relief="raised")
    Label1_6.configure(font="Pacifico")

    day = datetime.now().date().day
    month = datetime.now().date().month
    year = datetime.now().date().year
    date = str(day)+"-"+str(month)+"-"+str(year)

    log_list = [NAME,ID,date,str(TIME)[11:]]
    print(log_list)
    check1 = check_present([NAME,ID,date])

    if check1 == False:
        Label1_5.configure(text=str(TIME.hour)+":"+str(TIME.minute))
        Label1_6.configure(text='''Reading Temperature''')
        print("Going to countdown")
        countdown_temp(2)
    else:
        # log_len = log_length([NAME,date])
        # print("Log len = ",log_len)
        # if log_len == 4 or log_len == 6 or log_len == 7:
        #     # logging([NAME,date,TIME,0,TIME])
        log_list_out = [NAME,ID,date,TIME,0,TIME]
        Label1_5.configure(text="Already logged in")
        Label1_6.configure(text="""Reading Temperature""")
        countdown_temp1(3)
    

##    if temperature > 100:
##        Label1_6.configure(background="red")
##        Label1_6.configure(text="NOT OK")
##    else:
##        Label1_6.configure(background="green")
##        Label1_6.configure(text="OK")
    


def clear_window(window):
    for ele in window.winfo_children():
        ele.destroy()
def countdown(count):
    global top
        
    if count > 0:
        # call countdown again after 1000ms (1s)
        top.after(500, countdown, count-1)
    if count == 0:
        rekognise()
def countdown1(count1):
    global top
    if count1 > 0:
        top.after(1000, countdown1, count1-1)
    if count1 == 0:
        start(0)
def countdown_temp(count2):
    global top
    global Label1_6
    global log_list
    if count2 > 0:
        top.after(800, countdown_temp , count2-1)
    if count2 == 0:
        c = 0
        t = 0
        while True:
            if c==5:
                break

            temperature = temp()
            if temperature>92 and temperature<110:
                t+=temperature
                c+=1
            time.sleep(1)
        print(t/c)
        log_list += [t/c]
        if (t/c) > 100:
            Label1_6.configure(background = "red")
            Label1_6.configure(font=("Pacifico",20))
            Label1_6.configure(text=str(round(t/c,1)))
        else:
            Label1_6.configure(background = "green")
            Label1_6.configure(font=("Pacifico",20))
            Label1_6.configure(text=str(round(t/c,1)))
        write_log(log_list)
        countdown1(4)

def countdown_temp1(count3):
    global top
    global Label1_6
    global log_list_out
    if count3 > 0:
        top.after(800, countdown_temp1 , count3-1)
    if count3 == 0:
        c = 0
        t = 0
        while True:
            if c==5:
                break

            temperature = temp()
            if temperature>92 and temperature<110:
                t+=temperature
                c+=1
            time.sleep(1)
        print(t/c)
        log_list_out += [t/c]
        if (t/c) > 100:
            Label1_6.configure(background = "red")
            Label1_6.configure(font=("Pacifico",20))
            Label1_6.configure(text=str(round(t/c,1)))
        else:
            Label1_6.configure(background = "green")
            Label1_6.configure(font=("Pacifico",20))
            Label1_6.configure(text=str(round(t/c,1)))
        print(log_list_out)            
        write_log(log_list_out)
        countdown1(4)        
##########################################################################################
##########################################################################################

start()

# Release handle to the webcam
##video_capture.release()




