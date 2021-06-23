#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter import *
import pickle
from goldenapp3 import *


entries={}
entry={}
entry_var={}
errmsg = 'Error!'

def load_file():
     FILENAME=askopenfilename()
     try:
        ent = pickle.load( open( FILENAME, "rb" ) )
     except:
        ent={}

     for k in range(0,27):
         for j in range(0,6):
             try:
                 entries[k,j]=ent[k,j]
             except:
                 entries[k,j]=''

def save_file():
     FILENAME = asksaveasfilename()

     for k in range(0,27):
         for j in range(0,6):
             entries[k,j]=entry_var[k,j].get()

     pickle.dump(entries, open(FILENAME, "wb"))

     pickle.dump(entries, open('save.p', "wb"))

def populate():
     load_file()
     for j in range(0,6):
         entry_var[0,j] = tk.StringVar(root, entries[0,j])

     entry[0,0] = tk.Entry(root, width=10, textvariable=entry_var[0,0]).grid(row=0,column=1)
     entry[1,0] = tk.Entry(root, width=10, textvariable=entry_var[0,1]).grid(row=1,column=1)

     for k in range(1,27):
         for j in range(0,6):
             entry_var[k,j] = tk.StringVar(root, entries[k,j])
             entry[k,j] = tk.Entry(root, width=10, textvariable=entry_var[k,j]).grid(row=k+2,column=j+1)



# set the WM_CLASS
root = Tk(className="Goldenapp")
# set the window title
root.wm_title("GoldenApp Political Analytics Tool")


tk.Label(root, text="Country").grid(row=0)
tk.Label(root, text="Poll").grid(row=1)
tk.Label(root, text="Party Name").grid(row=2, column=1)
tk.Label(root, text="Party Label").grid(row=2, column=2)
tk.Label(root, text="Party Color").grid(row=2, column=3)
tk.Label(root, text="Seats Proportion").grid(row=2, column=4)
tk.Label(root, text="Excluded Partners").grid(row=2, column=5)
tk.Label(root, text="Line in Excel").grid(row=2, column=6)
tk.Label(root, text="A").grid(row=3)
tk.Label(root, text="B").grid(row=4)
tk.Label(root, text="C").grid(row=5)
tk.Label(root, text="D").grid(row=6)
tk.Label(root, text="E").grid(row=7)
tk.Label(root, text="F").grid(row=8)
tk.Label(root, text="G").grid(row=9)
tk.Label(root, text="H").grid(row=10)
tk.Label(root, text="I").grid(row=11)
tk.Label(root, text="J").grid(row=12)
tk.Label(root, text="K").grid(row=13)
tk.Label(root, text="L").grid(row=14)
tk.Label(root, text="M").grid(row=15)
tk.Label(root, text="N").grid(row=16)
tk.Label(root, text="O").grid(row=17)
tk.Label(root, text="P").grid(row=18)
tk.Label(root, text="Q").grid(row=19)
tk.Label(root, text="R").grid(row=20)
tk.Label(root, text="S").grid(row=21)
tk.Label(root, text="T").grid(row=22)
tk.Label(root, text="U").grid(row=23)
tk.Label(root, text="V").grid(row=24)
tk.Label(root, text="W").grid(row=25)
tk.Label(root, text="X").grid(row=26)
tk.Label(root, text="Y").grid(row=27)
tk.Label(root, text="Z").grid(row=28)

populate()

entry[28] = tk.Button(root, text='Load', command= lambda:populate()).grid(row=30,column=0)
entry[29] = tk.Button(root, text='Save', command= lambda:save_file()).grid(row=30,column=1)
entry[30] = tk.Button(root, text='Run', command= lambda:goldenapp()).grid(row=30,column=2)
entry[31] = tk.Button(root, text='Quit', command= root.quit).grid(row=30,column=3)

root.mainloop()



