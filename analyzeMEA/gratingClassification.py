import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate
from PIL import Image, ImageTk
from tkinter import *
from tkinter.font import Font
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk)


def classify_gratings_manually(footfallImages,savename = 'grating_classification.npy', pawPositions=None):


    window = Tk()
    window.title('Classify Orientations')
    window.geometry('800x400')
    window.resizable(0,0)
    try:
        grating_classification = np.load(savename)
        grating_classification = [n for n in grating_classification]
    except:
        print('No gratings file found, starting new one')
        grating_classification = []
    numFrames = len(footfallImages[len(grating_classification):]) ## add this value to display
    totalFrames = len(footfallImages)
    global prevFrames
    prevFrames = len(grating_classification)
    


    ## making images for buttons
    sineWave = np.uint8(255*(np.sin(np.arange(0,100,0.1))/2+0.5))[::10]
    vertImage = np.matlib.repmat(sineWave,len(sineWave),1)
    horizImage = vertImage.T
    rightDiagImage = rotate(vertImage,angle=-45,reshape=False)
    leftDiagImage = rotate(vertImage,angle=45,reshape=False)


    def _photo_image(image: np.ndarray):
        height, width = image.shape
        data = f'P5 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
        return PhotoImage(width=width, height=height, data=data, format='PPM')

    vertImage = _photo_image(vertImage)
    horizImage = _photo_image(horizImage)
    rightDiagImage = _photo_image(rightDiagImage)
    leftDiagImage = _photo_image(leftDiagImage)

    
    
    def plot(im):
    ## plot image of mouse and paw positions      
            fig = plt.figure(figsize=(4,4),dpi=100)
            ax = plt.axes()
            ax.imshow(im)
            if pawPositions is not None:
                ax.plot(pawPositions[prevFrames,0],pawPositions[prevFrames,1],'.',color='r')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Frame {} of {}'.format(prevFrames, totalFrames))
            canvas = FigureCanvasTkAgg(fig, master = window)
            canvas.draw()
            canv = canvas.get_tk_widget()
            canv.grid(row = 0, column = 0,
       columnspan = 3, rowspan = 3, padx = 3, pady = 3)
            plt.close()
            return canv
            # toolbar = NavigationToolbar2Tk(canvas, window)
            # toolbar.update()
            # canvas.get_tk_widget().pack(side='left')
    

    def vertButtonClick():
        grating_classification.append(4)
        global prevFrames, canv
        np.save(savename,np.int32(grating_classification))
        prevFrames += 1
        canv.grid_forget()
        try:
            canv = plot(footfallImages[prevFrames])
        except IndexError:
            print('Congratulations, you have reached the end')
            window.quit()
        

    def rightDiagButtonClick():
        grating_classification.append(3)
        global prevFrames, canv
        np.save(savename,np.int32(grating_classification))
        prevFrames += 1
        canv.grid_forget()
        try:
            canv = plot(footfallImages[prevFrames])
        except IndexError:
            print('Congratulations, you have reached the end')
            window.quit()
        

    def horizButtonClick():
        grating_classification.append(2)
        np.save(savename,np.int32(grating_classification))
        global prevFrames, canv
        prevFrames += 1
        canv.grid_forget()
        try:
            canv = plot(footfallImages[prevFrames])
        except IndexError:
            print('Congratulations, you have reached the end')
            window.quit()
        
    def leftDiagButtonClick():
        grating_classification.append(1)
        np.save(savename,np.int32(grating_classification))
        global prevFrames, canv
        prevFrames += 1
        canv.grid_forget()
        try:
            canv = plot(footfallImages[prevFrames])
        except IndexError:
            print('Congratulations, you have reached the end')
            window.quit()
        

    def NAButtonClick():
        grating_classification.append(0)
        np.save(savename,np.int32(grating_classification))
        global prevFrames, canv
        prevFrames += 1
        canv.grid_forget()
        try:
            canv = plot(footfallImages[prevFrames])
        except IndexError:
            print('Congratulations, you have reached the end')
            window.quit()
        
    
    def UndoButtonClick():
        grating_classification.pop()
        np.save(savename,np.int32(grating_classification))
        global prevFrames, canv
        prevFrames -= 1
        canv.grid_forget()
        try:
            canv = plot(footfallImages[prevFrames])
        except IndexError:
            print('Congratulations, you have reached the end')
            window.quit()
        
    
    vertButton = Button(master = window,
                        height = 75,
                        width = 75,
                        image = vertImage,
                        command= vertButtonClick)

    rightDiagButton = Button(master = window,
                        height = 75,
                        width = 75,
                        image = rightDiagImage,
                        command= rightDiagButtonClick)

    horizButton = Button(master = window,
                        height = 75,
                        width = 75,
                        image = horizImage,
                        command = horizButtonClick)

    leftDiagButton = Button(master = window,
                        height = 75,
                        width = 75,
                        image = leftDiagImage,
                        command= leftDiagButtonClick)

    NAButton = Button(master = window,
                        height = 4,
                        width=5,
                        text='NONE',
                        background='red',
                        command=NAButtonClick)

    UndoButton = Button(master = window,
                        height = 4,
                        width = 5,
                        text = 'UNDO',
                        background='white',
                        command = UndoButtonClick)
    Desired_font = Font( family = "Arial", 
                                 size = 16, 
                                 weight = "bold")
    l6 = Label(master = window, text = "(6)",font=Desired_font)
    l7 = Label(master = window, text = "(7)",font=Desired_font)
    l8 = Label(master = window, text = "(8)",font=Desired_font)
    l9 = Label(master = window, text = "(9)",font=Desired_font)
    l0 = Label(master = window, text = "(0)",font=Desired_font)
    back = Label(master = window, text= "(back)")
    instructions = Label(master = window, text = 
                         "Match texture at location of \nright forepaw to button image.",font=Desired_font)
    l6.grid(row = 1, column = 3, sticky = S)
    l7.grid(row = 1, column = 4, sticky = S)
    l8.grid(row = 1, column = 5, sticky = S)
    l9.grid(row = 1, column = 6, sticky = S)
    l0.grid(row = 1, column = 7, sticky = S)
    back.grid(row=0, column = 7, sticky = S)
    instructions.grid(row = 0, column = 3, columnspan = 4)


    global canv
    canv = plot(footfallImages[prevFrames])
    

    

    horizButton.grid(row = 1, column = 3, padx = 2, pady = 2)
    leftDiagButton.grid(row = 1, column = 4, padx = 2, pady = 2)
    vertButton.grid(row = 1, column = 5, padx = 2, pady = 2)
    rightDiagButton.grid(row = 1, column = 6, padx = 2, pady = 2)
    UndoButton.grid(row = 0, column = 7)
    NAButton.grid(row = 1, column = 7, padx = 2, pady = 2)
    
    window.bind('6',lambda event: horizButtonClick())
    window.bind('7',lambda event: leftDiagButtonClick())
    window.bind('8',lambda event: vertButtonClick())
    window.bind('9',lambda event: rightDiagButtonClick())
    
    
    window.bind('0',lambda event: NAButtonClick())
    window.bind('<BackSpace>', lambda event: UndoButtonClick())

    menubar = Menu(window)
    
    
    file = Menu(menubar, tearoff=0)  
    file.add_command(label="New")  
    file.add_command(label="Open")  
    file.add_command(label="Save")  
    file.add_command(label="Save as...")  
    file.add_command(label="Close")  
    file.add_separator()  
    file.add_command(label="Exit", command=window.quit)


    menubar.add_cascade(label="File", menu=file)  
    edit = Menu(menubar, tearoff=0)  
    edit.add_command(label="Undo",command=UndoButtonClick)
    menubar.add_cascade(label="Edit", menu=edit)  
    help = Menu(menubar, tearoff=0)  
    help.add_command(label="About")  
    menubar.add_cascade(label="Help", menu=help)
  
    window.config(menu=menubar)

    window.mainloop()




def main():
    initials = input("what are your initials?")
    root = Tk()
    root.update()
    images_filename = askopenfilename(defaultextension='.npy',title='Select file with footfall images')
    root.update()
    positions_filename = askopenfilename(defaultextension='.npy',title='Select file with paw positions')
    root.destroy()
    classify_gratings_manually(np.load(images_filename),os.path.split(images_filename)[0]+'/gratingClassifications_'+initials+'.npy',
                                pawPositions=np.load(positions_filename))

if __name__ == "__main__":
    main()