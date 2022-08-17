import speech_recognition as sr     # import the library
import tkinter as tk
from generate import generate_unconditionally, generate_conditionally


def SpeakFunc():
	r = sr.Recognizer()                 # initialize recognizer
	with sr.Microphone() as source:     # mention source it will be either Microphone or audio files.
		print("Speak Anything :")
		audio = r.listen(source)        # listen to the source
	try:
	    text = r.recognize_google(audio)    # use recognizer to convert our audio into text part.
	    print("You said : {}".format(text))
	    generate_conditionally(text)
	except:
		print("Sorry could not recognize your voice")

def quit():
	root.destroy()


root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, text="speak", command=SpeakFunc)
button.pack()
button_quit = tk.Button(frame, text="quit", command=quit)
button_quit.pack()

root.geometry('500x500')
root.mainloop()