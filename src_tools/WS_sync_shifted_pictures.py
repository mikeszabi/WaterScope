
import PIL
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt


'''The two main functions correctRGB_shift_frompath and correctRGB_shift take
the path of the image or the 3 x tuple/list (RGB) as input and returns a shift 
corrected numpy array.

Tested on 3000 images, run without errors. In average 49% had 0 pixel shift 32%
had 1, 14% 2, 2.7% 3. After correction approx 95% images showed no shift after 
remeasuring and 1-1.5% had more than 1 pixel shift remaining.

The algorithm chooses the center of all 3 colors based on the darkest 1% pixels.
The color with the closest center to the image center is choosen as central layer.
The linear color transformation parameters are calculated between the central 
image and the two others. A subjective window (10 pixels) is taken and the sum 
error (sum(img(i,j)*a+b-img2(i+x,j+y))) minimized, where x,y is the translation.
The optimum is searched by greedy (gradient) method. 9 errors are calculated 
(central point+8 neighbor pixels), the lowest choosen, missing neighbour pixel
errors calculated, and repeated until local minimum is found. Afterwards images 
are croped together according to the calculated translations.

Contains extra outputs, which are disabled by default.

'''


#supplementary functions
def list_dir(dir_name, traversed = [], results = []): 
    dirs = os.listdir(dir_name) #files and folders
    dir_name=dir_name+'/'
    if dirs:
        for f in dirs:
            new_dir = dir_name + f+'/'   # tries to access everything
            if os.path.isdir(new_dir) and new_dir not in traversed: #if it is a directory and we havent been there yet
                traversed.append(new_dir) #add to been there map
                list_dir(new_dir, traversed, results) #checks whats in the dir
            else:
                if new_dir[-4:-1]=='png' or new_dir[-4:-1]=='PNG'or new_dir[-4:-1]=='jpg' or new_dir[-4:-1]=='JPG':
                    results.append([new_dir[:-1]])
    return results


def list_of_files_in_dir(dir_name):
    '''List pictures in dir   
    
    Can be called as: 
    fileok=list_of_files_in_dir(dir)
    for i in fileok:
        path=str(i)[2:-2]
        img=formatimage(path)    
    '''

    fileok=[]
    for file_name in list_dir(dir_name):
         print(file_name) # sample with file size
         fileok.append(file_name)
    return fileok

def save_image_numpy(img,path,im_name,show=0):
    '''Input name should contain .png extension
    '''    
    if show !=0:        
        plt.imshow(img)
        plt.show()
    image = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
    image.save(path+'\\'+im_name,"PNG")
    
    return

def save_image_tuple(img_tup,path,im_name,show=0):
    img_numpy=np.zeros((img_tup[0].shape[0],img_tup[0].shape[1],3)) #numpy
    img_numpy[:,:,0]=img_tup[0]
    img_numpy[:,:,1]=img_tup[1]
    img_numpy[:,:,2]=img_tup[2]
    
    save_image_numpy(img_numpy,path,im_name,show)
    
    return


def create_channels_tuple(im):
    assert type(im)==np.ndarray, "Not numpy image"
    assert im.ndim==3, "Not 3 channel image"

    imgRed=np.asarray(im[:,:,0],dtype=np.uint8)
    imgGreen=np.asarray(im[:,:,1],dtype=np.uint8)
    imgBlue=np.asarray(im[:,:,2],dtype=np.uint8)
    #optional streching
#    imgRed=imgRed-np.min(imgRed)  #stretch to 0
#    imgGreen=imgGreen-np.min(imgGreen)
#    imgBlue=imgBlue-np.min(imgBlue)

    #optional
#    imgRed=np.round(float(imgRed)/np.max(imgRed)*255) #stretch to 255
#    imgGreen=np.round(imgGreen/np.max(imgGreen)*255)
#    imgBlue=np.round(imgBlue/np.max(imgBlue)*255)


    return imgRed,imgGreen,imgBlue

##Actual functions
def formatimage(path):
    '''Takes path, returns 3 x tuple RGB
    '''
    
    img=PIL.Image.open(path)  
    img_rgb=img.convert('RGB')     # if png remove 4th layer 3dim
    im = np.asarray(img_rgb,dtype=np.uint8) #numpy it     
    return create_channels_tuple(im)


def ab_param(img,img2):
    '''Analytically calculate color a,b parameter from two images
    '''
    
    #input first x, second y
    # data in order X2, X, XY, Y, Y2
    img_combo=np.ones((img.shape[0]*img.shape[1],5)) #fill with ones
    img_flat=img.flatten() #get x values
    img2_flat=img2.flatten() #get y values
    img_combo[:,0]=img_combo[:,0]*img_flat*img_flat #xx
    img_combo[:,1]=img_combo[:,1]*img_flat#x
    img_combo[:,2]=img_combo[:,2]*img_flat*img2_flat#xy
    img_combo[:,3]=img_combo[:,3]*img2_flat#y
    img_combo[:,4]=img_combo[:,4]*img2_flat*img2_flat#yy
    
    #sum each parameter
    sumxy=np.array([
                    np.sum(img_combo[:,0]), #sumxx
                    np.sum(img_combo[:,1]), #sumx
                    np.sum(img_combo[:,2]), #sumxy
                    np.sum(img_combo[:,3]), #sumx
                    np.sum(img_combo[:,4])  #sumxx
                    ])   
    av_sumxy=sumxy/img_flat.shape[0] #take average
    
    
    #analytic calc a,b parameters
    #two identic equation sets
    
    #a=(av_sumxy[2]-av_sumxy[1]*av_sumxy[3])/(av_sumxy[0]-av_sumxy[1]*av_sumxy[1])
    #b=av_sumxy[3]-a*av_sumxy[1]
    b=(av_sumxy[3]* av_sumxy[0]-  av_sumxy[2]*av_sumxy[1])/(av_sumxy[0]-av_sumxy[1]*av_sumxy[1])
    a=(av_sumxy[2]-b*av_sumxy[1])/av_sumxy[0]
    
    t=(a,b)
    return t


def error_calc(t,img,img2,pixels=10,x=0,y=0):
    '''Error calculation between two images
    '''

    #pixels=how much smaller is the moving image
    #x,y=current dislocation in pixels
    #errormap is an option, not needed always
    
    error=0
    #errormap=np.zeros((img.shape[0]-2*pixels,img.shape[1]-2*pixels))
    for i in range(pixels,img.shape[0]-pixels): #only iterate on floating top
        for j in range(pixels,img.shape[1]-pixels):
     #       errormap[i-pixels,j-pixels]=np.square(t[0]*float(img[i,j])+t[1]-float(img2[i+x,j+y])) #img2 sliding over
     #       error=error+errormap[i-pixels,j-pixels]         
    
             error=error+np.square(t[0]*float(img[i,j])+t[1]-float(img2[i+x,j+y]))
    c,d=img.shape
    error=error/(c*d) #norm by pixel amount
    
    return error


def del_row_column(img,row,column):
    '''Takes positive, negative numbers and 0 as input
    '''
    
    a=0
    if row <0:
       a=-1 
    for _ in range(abs(row)):
        img=np.delete(img,a,0)
    b=0
    if column <0:
       b=-1 
    for _ in range(abs(column)):
        img=np.delete(img,b,1)
    return img


def fitting_Greedy(t,pixels,img_center,img_rel):
    '''Calculating dislocation with Greedy(gradient method)
    
    Takes in (a,b) param, pixel window, center and relativ picture
    Returns dislocation row,column
    '''     
    
    #Initialize
    lim=10000000000 #Big number, because minimum will be searched 
    er_values=np.ones((2*pixels+1,2*pixels+1))*lim #generate error field        
    A=np.array([-1,0,1]) #help to make the iterator
    center_point=(0,0)
    cold=center_point
    
    #the main cycle 
    while True:       
        
        #calc error 1 pixel around, 9 alltogether    
        for i, j in itertools.product(A, A): 
            if er_values[i+pixels+cold[0], j+pixels+cold[1]] ==lim:
                er=error_calc(t,img_center,img_rel,pixels,i+cold[0],j+cold[1])
                er_values[i+pixels+cold[0],
                          j+pixels+cold[1]]=er
            
        #get minimum inside of the 3x3 matrix
        cnew= np.unravel_index( #reshape index
                np.argmin( #get min in 9 pixels
                        er_values[pixels+cold[0]-1:pixels+cold[0]+2,pixels+cold[1]-1:pixels+cold[1]+2]) #around the previous match 
                ,(3,3)) #reshape to 3x3           
                          
        cnew=(cold[0]+cnew[0]-1,cold[1]+cnew[1]-1) #get absolute values
        if cnew==cold: #found optimum leaving
            break
        else:
            cold=cnew
        if abs(cold[0])==pixels or abs(cold[1])==pixels: #run out of pixels
            break

    disloci=cold[0] #calc dislock   
    dislocj=cold[1] #calc dislock   

    return disloci,dislocj


def cropRGB(img_center,img_rel1,img_rel2,dislocrow1,disloccol1,dislocrow2,disloccol2):
    '''Crop RGB image together
    
    Takes three two dimension images with 4 (two row and two column) dislocation values. 
    Returns three overlapping images in input order, 
    plus row/column dislocation measurement between the two moving images, 
    which should be zero if input values were correct (not always).
    
    Input:img_center,img_rel1,img_rel2,dislocrow1,disloccol1,dislocrow2,disloccol2
    Return:img_center_mod,img_rel1_mod,img_rel2_mod,dislocrow,disloccol
    '''
    
    #debugged, working
    #sort images
    if abs(dislocrow1) < abs(dislocrow2): #make sure dislocrow2 is bigger in abs value
        switch=0
        img_big=img_rel2.copy()
        img_small=img_rel1.copy()
        big=dislocrow2
        small=dislocrow1
    else:
        switch=1
        img_big=img_rel1.copy()
        img_small=img_rel2.copy()
        big=dislocrow1
        small=dislocrow2
    
    #modify rows    
    if dislocrow1*dislocrow2>0: #make sure they are pointing in same direction
       img_big=del_row_column(img_big,-big,0)
       img_center=del_row_column(img_center,big,0) #the bigger in abs value will dominate       
       if big==small:
           img_small=del_row_column(img_small,-big,0) #same step same direction, one side crop
       else:
           img_small=del_row_column(img_small,big-small,0) #has to be cropped both from beg and end
           img_small=del_row_column(img_small,-small,0)    
    else: #either oposite direction or one of them is zero
       img_big=del_row_column(img_big,-big+small,0)
       img_center=del_row_column(img_center,big,0)
       img_center=del_row_column(img_center,small,0) 
       img_small=del_row_column(img_small,big-small,0)
         
              
    #switch back, only images count
    if switch==0:
        img_rel2=img_big.copy()
        img_rel1=img_small.copy()
    else:
        img_rel1=img_big.copy()
        img_rel2=img_small.copy()
    
    #do the same for columns
    #sort images
    if abs(disloccol1) < abs(disloccol2): #make sure dislocrow2 is bigger in abs value
        switch=0
        img_big=img_rel2.copy()
        img_small=img_rel1.copy()
        big=disloccol2   #
        small=disloccol1
    else:
        switch=1
        img_big=img_rel1.copy()
        img_small=img_rel2.copy()
        big=disloccol1
        small=disloccol2
    
    #modify cols    
    if disloccol1*disloccol2>0: #make sure they are pointing in same direction
       img_big=del_row_column(img_big,0,-big)
       img_center=del_row_column(img_center,0,big) #the bigger in abs value will dominate       
       if big==small:
           img_small=del_row_column(img_small,0,-big)
       else:
           img_small=del_row_column(img_small,0,big-small)
           img_small=del_row_column(img_small,0,-small)     
    else: #either oposite direction or one of them is zero
       img_big=del_row_column(img_big,0,-big+small)
       img_center=del_row_column(img_center,0,big)
       img_center=del_row_column(img_center,0,small) 
       img_small=del_row_column(img_small,0,big-small)
        
    #switch back, only images count
    if switch==0:
        img_rel2=img_big.copy()
        img_rel1=img_small.copy()
    else:
        img_rel1=img_big.copy()
        img_rel2=img_small.copy()
           
    return img_center,img_rel1,img_rel2


def image_center(img, fraction):
    '''Calculates image center based on fraction of darkest points
    
    fraction under 0.01 seems a good choice    
    '''
    
    hist=np.histogram(img, bins='auto', normed=False, weights=None, density=False)
  
    #choose the darkest fraction of the pixels
    i=0
    area=0 #pixel amount
    while area<fraction*(img.shape[0]*img.shape[1]): 
        area +=hist[0][i]
        i+=1
    background=hist[1][i] #the 0-255 value under which the pixel counts       
   
    centerpointx=img.shape[0]/2 #row
    centerpointy=img.shape[1]/2 #col
    x, y = (img < background).nonzero() # points darker than background    
    image=background-img # the diff will be needed for weight calc    
    weights2 = image[x, y] # get pixel darkness
    weights =weights2/np.sum(weights2) #norm

    #weighted normed coordinates
    x=np.sum((x-centerpointx)*weights)
    y=np.sum((y-centerpointy)*weights)
    return x,y

def calculate_image_dislocation(img,pixels=10,img2=0,center=7):
    '''Takes image tuple and pixel window, optionally image for A/B calculation, returns dislocation, center image id
    '''
    
    if center==7: #if center not sepecified calculate
        distparam=np.zeros((3,3)) #contains image center x,y coordinates and distance from image center
        for i in range(3):
            a,b=image_center(img[i],0.01)
            distparam[i,:]=[a,b,np.sqrt(a*a+b*b)]
               
        center=np.where(distparam[2,:]==min(distparam[2,:]))[0][0] #choose center from RGB        
    
    #direction will be always R-G-B
    disloc=np.zeros((3,2)) #contains all dislocations
    for i in range(3):
        if i==center:
            continue
        else:
            if img2==0:
                img2=img
            t=ab_param(img2[center],img2[i]) # a and b color parameters, you can use diff pic for calculation 
            disloc[i,0],disloc[i,1]=fitting_Greedy(t,pixels,img[center],img[i])
    return disloc,center


def correctRGB_shift(img,pixels=10):
    '''Takes in 3 x tuple/list (RGB) from formatimage, returns shift corrected numpy image
    
    Optionally two corrections can be done 
    '''
    
      
    if min(img[0].shape[0],img[0].shape[1])<2*pixels+5: #too small image, return original image
        
        img_mod_sorted=np.zeros((img[0].shape[0],img[0].shape[1],3))
        img_mod_sorted[:,:,0]=img[0]
        img_mod_sorted[:,:,1]=img[1]
        img_mod_sorted[:,:,2]=img[2]
        img_mod_sorted=np.asarray(img_mod_sorted,dtype=np.uint8)
#        data='found nothing'
#        data2='found nothing'

    else:
        disloc_first,center=calculate_image_dislocation(img,pixels)
        print(disloc_first)
                    
        A=np.array([0,1,2]) #RGB color ids
        B=np.delete(A,np.where(A==center)[0],0) #center removed, contains relative image indexes in order   
        disloc=np.delete(disloc_first,center,0) #remove 0s at central images place
#        img_org=img    
        img_mod=cropRGB( #crop images accprding to shift
                img[center],img[B[0]],img[B[1]],  #center and two shifted images
                #has to be negative to compensate for dislocation
                int(-disloc[0,0]),int(-disloc[0,1]),
                int(-disloc[1,0]),int(-disloc[1,1])
                )
        
        #restore original layer order in list
        img=[[],[],[]]
        img[center]=img_mod[0]
        img[B[0]]=img_mod[1]
        img[B[1]]=img_mod[2]
        img=tuple(img)
        
        #measures dislocs again
#        disloca,center=calculate_image_dislocation(img,pixels) #from mod image
#        if np.sum(disloca)!=0: #only do this if there is still shift
#            
#            dislocb,center=calculate_image_dislocation(img,pixels,img_org) # using orig a,b param
#            dislocc,center=calculate_image_dislocation(img,pixels,img_org,center) # using orig ab and initial center
#            dislocd,center=calculate_image_dislocation(img,pixels,img,center) # using mod with initial center
#            
#            data=np.zeros((3,10))
#            data[:,0:2]=disloc_first
#            data[:,2:4]=disloca
#            data[:,4:6]=dislocb
#            data[:,6:8]=dislocc
#            data[:,8:10]=dislocd
            
#            #do another cut
#            B=np.delete(A,np.where(A==center)[0],0) #center removed, contains relative image indexes in order   
#            disloc=np.delete(disloca,center,0) #remove 0s at central images place
#            img_mod=cropRGB( #crop images accprding to shift
#                img[center],img[B[0]],img[B[1]],  #center and two shifted images
#                #has to be negative to compensate for dislocation
#                int(-disloc[0,0]),int(-disloc[0,1]),
#                int(-disloc[1,0]),int(-disloc[1,1])
#                )
#            #restore original layer order in list
#            img=[[],[],[]]
#            img[center]=img_mod[0]
#            img[B[0]]=img_mod[1]
#            img[B[1]]=img_mod[2]
#            img=tuple(img)
#            
            #measure the distances the same way as before, use the same names as important values saved out
#            disloca,center=calculate_image_dislocation(img,pixels) #from mod image
#            if np.sum(disloca)!=0: #only do this if there is still shift
#            
#                dislocb,center=calculate_image_dislocation(img,pixels,img_org) # using orig a,b param
#                dislocc,center=calculate_image_dislocation(img,pixels,img_org,center) # using orig ab and initial center
#                dislocd,center=calculate_image_dislocation(img,pixels,img,center) # using mod with initial center
#                
#                data2=np.zeros((3,10))
#                data2[:,0:2]=disloc_first
#                data2[:,2:4]=disloca
#                data2[:,4:6]=dislocb
#                data2[:,6:8]=dislocc
#                data2[:,8:10]=dislocd
#            else:
#                data2='found nothin'               
#        else:
#            data='found nothin'
#            data2='found nothin'
        
        # numpy        
        img_mod_sorted=np.zeros((img[0].shape[0],img[0].shape[1],3))
        img_mod_sorted[:,:,0]=img[0]
        img_mod_sorted[:,:,1]=img[1]
        img_mod_sorted[:,:,2]=img[2]
        img_mod_sorted=np.asarray(img_mod_sorted,dtype=np.uint8)
        
    return img_mod_sorted


def correctRGB_shift_frompath(path):
    '''Same as correctRGB_shift just works directly from path
    '''
    
    img=formatimage(path)
    img_mod_sorted=correctRGB_shift(img)
    return img_mod_sorted

  


