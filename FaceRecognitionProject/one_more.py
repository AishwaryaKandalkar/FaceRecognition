from tkinter import *  
# import Login as lg
# import MoreTry as mt
  
top = Tk()  
top.title("Harshal")
top.geometry("400x400")  
  
def login():
    id = Label(top, text = "user_id").place(x = 30,y = 50)  
    e1 = Entry(top).place(x = 80, y = 50)  
    e1.pack()  
    b2 = Button(text="camera").place(x=80,y=110)
    b2.pack()
    
def start_camera():
    c = Label(top, text = "user_id").place(x = 30,y = 50) 
  

  
# build = Button(top, text = "build data",activebackground = "pink", activeforeground = "blue").place(x = 30, y = 90)  
  
  
b1 = Button(top, text = "Login",activebackground = "pink", activeforeground = "blue", command=login).place(x = 30, y = 170)  
  
# start_test = Button(top, text = "start test",activebackground = "pink", activeforeground = "blue").place(x = 30, y = 300)  

# start_test.pack() 
# e2 = Entry(top).place(x = 80, y = 90)  
  
  
# e3 = Entry(top).place(x = 95, y = 130)  
  
top.mainloop()  