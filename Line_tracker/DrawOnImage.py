from tkinter import Canvas, Frame, Tk
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt




class Drawing():
    def __init__(self):
        self.Coord = []
    
    def add(self, Coord):
        self.Coord.append(Coord)


class Draw_Application(Frame):
    def __init__(self, img):
        
        if type(img) is str:
            self.image = Image.open(img)
        elif type(img) is np.ndarray:
            if img.max() == 1:
                self.image = Image.fromarray(img*255)
            else:
                self.image = Image.fromarray(img)
        size = np.array(self.image.size)
        maxsize = [1500, 800]
        self.ratio = min(maxsize[0]/size[0], maxsize[0]/size[1])
        self.new_size = (np.rint(size*self.ratio)).astype(int)
        
        self.root = Tk()
        #self.root.attributes('-fullscreen',1)
        text_size = '%ix%i' %tuple(self.new_size)
        self.root.geometry(text_size)
        self.root.bind('<Return>', lambda event: self.root.destroy())
        Frame.__init__(self, self.root)
        #self.Coord = np.zeros((2,0), 'int32')
        self.Coord = []
        self.Drawings = []
        self.create_widgets()



    def create_widgets(self):

        self.canvas = Canvas(self.root, bg='black')
        self.canvas.pack(anchor='nw', fill='both', expand=1)

        self.canvas.bind("<Button-1>", self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.action)
        print('ready')
        
        NewDrawing = Drawing()

        NewDrawing.add(self.Coord)
        self.Drawings.append(NewDrawing)
        self.image = self.image.resize(self.new_size, Image.LANCZOS)
        #self.image = Image.open("frame_0002.png")
        #self.image = self.image.resize((800,800), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0,0, image=self.image, anchor='nw')
     
        
    def action(self, event):
        #print('init')
        #self.Coord = []
        self.draw_smth(event)
        
    def get_x_and_y(self, event):
        self.lasx, self.lasy = event.x, event.y

    def draw_smth(self,event):
        #self.Coord=[]
        self.canvas.create_line((self.lasx, self.lasy, event.x, event.y), fill='red', width=2)
        self.lasx, self.lasy = event.x, event.y
        #self.Coord = np.append(self.Coord, np.array([self.lasx,self.lasy]))
        self.Coord.append(np.array([self.lasx,self.lasy]))    

        
        
    def parse(self, event):
        print("You clicked?")

    def start(self):
        self.root.mainloop()
        self.convert()
        return self.Line
    
    def convert(self):
        line = np.array(self.Coord, 'int32')/self.ratio
        line = (np.rint(line)).astype(int)
        self.Line = line.reshape(line.shape[0],1,2)
        


if __name__ == '__main__':

    img = 'images/frame_0002.png'
    a= Draw_Application(img).start()
    # drawing = Draw_Application()
    # drawing.start()
    frame = cv2.imread(img)
    line = np.array(a[0], 'int32')/a[1]
    line = (np.rint(line)).astype(int)
    line = line.reshape(line.shape[0],1,2)
    fig, ax = plt.subplots()
    cv2.drawContours(frame, line, -1, (0, 255, 0), 3)
    print('hi')