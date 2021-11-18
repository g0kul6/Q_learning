import glob
from PIL import Image

files=glob.glob('frames//*.png')
frames=[]
num=[]
path=[]
for i in files:
    a=int(i.split("\\")[1].split('.')[0])
    path=i.split('.')[0][0:7]
    num.append(a)
num.sort()
for i in num:
    a=path+str(i)+'.png'
    img=Image.open(a)
    img=img.resize((200,200))
    
    frames.append(img)
frames[0].save('taxi.gif', format='GIF',append_images=frames[0:],save_all=True,duration=700, loop=0)