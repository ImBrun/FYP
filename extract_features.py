import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage import morphology
import cv2
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import extcolors
from colormap import rgb2hex
import os
import matplotlib.colors as colors
import skimage
import skimage.segmentation as segmentation
from skimage.measure import regionprops
from skimage.color import rgb2hsv
from collections import Counter
from skimage.color import rgb2gray
import numpy.ma as ma


#-----------Asymmetry feature code--------------


def nonzero_index(array, first=True):
    # If first is set to true, return the index of the first non zero element
    # If first is set to false, return the index of the last non zero element 
    
    fnzi = -1 
    indices = np.flatnonzero(array) #gives an array of the indices of the non zero elements
       
    if (len(indices) > 0):
        if first:
            fnzi = indices[0] #index of first non zero element
        else:
            fnzi = indices[-1] # index of last non zero element
        
    return fnzi


def crop_rectangle_mask(mask):
    # Crop the mask to the exact rectangle needed to contain it, getting rid of any unneccessary black space
    
    width = None
    height = None
    
    first = True
    
    rightmost_col = None; leftmost_col = None; highest = (mask.shape)[1]; lowest = 0;
    

    for column in range(mask.shape[1]):
            
        col = mask[:, column]
        #if column is all black skip
        if not np.any(col):
            continue
                
        else:
            #if its the first not all black columnn the leftmost column where the lesion is is this
            if first:
                leftmost_col = column
                first = False
            index_first = nonzero_index(col, first=True)
            index_last = nonzero_index(col, first=False)
            #get highest point lesion reaches (row index)
            if index_first < highest:
                highest = index_first
            #get lowest point lesion reaches (row index)
            if index_last > lowest:
                lowest = index_last
            
            rightmost_col = column
    
    mask_cropped = mask[highest:(lowest + 1), leftmost_col:(rightmost_col)+1]
                
    return mask_cropped

# The mask input has to be resized
def asymmetry(mask):
    
    mask=crop_rectangle_mask(mask)
    if mask.shape[0]%2 != 0:
        #add row of zeros
        mask = np.append(mask, np.zeros((1,mask.shape[1]),dtype=int), axis=0)
    if mask.shape[1]%2 != 0:
        #add column of zeros
        mask = np.append(mask, np.zeros((mask.shape[0],1),dtype=int), axis=1)
    height_mask=mask.shape[0]
    width_mask=mask.shape[1]
    
    #divides the halves
    center_row = height_mask//2
    center_col = width_mask//2
    left = mask[:,:center_col]
    right = mask[:,center_col:]
    top = mask[:center_row, :]
    bottom = mask[center_row:, :]
    #flips one side to fit the other so we can compare
    r_to_left = cv2.flip(right, 1)
    vert_asym = left+r_to_left
    vert_difference = sum(vert_asym[vert_asym==1])
    top_to_b = cv2.flip(top, 0)
    hor_asym = bottom+top_to_b
    hor_diff = sum(hor_asym[hor_asym==1])
    asymmetry_score = (vert_difference+hor_diff)/np.sum(mask)*100
    return np.around(asymmetry_score, decimals=3)

#-------------------------------------------------------

#------------Border irregularity code---------------

# The higher the comp. value -> the more uneven the border
def compactness(mask):
    
    mask_img = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
       
    #rounding values in image array so that it is fully binary - consisting only of zeros and ones
    mask_img = np.around(mask_img)

    # computing the area of the lesion - the sum of all the ones in the array
    area = np.sum(mask_img)
    
    #creating a structuring element that will be used as a brush to "eat away" the otlining pixels of the lesion
    str_el = morphology.disk(1)
    eroded = morphology.binary_erosion(mask_img, str_el)
    
    #computing the perimeter - finding the number of 1s in the outline of the lesion
    perimeter_outline = mask_img - eroded
    perimeter = np.sum(perimeter_outline)
    
    #Compute the compactness according to the formula perimeter squared devided by the area times 4 pi
    comp = (perimeter*perimeter)/(4*(np.pi)*area)
     
    return comp

#-----------------------------------------------------

#-------------Colors features code--------------------

def get_hsv_vals(img, mask):
    
    SEG_COUNT = 150
    
    segments_slic = segmentation.slic(img[:, :, :3], n_segments=SEG_COUNT, mask=mask, compactness=3, sigma=3, start_label=1)
    
    props = regionprops(segments_slic, intensity_image=img)
    h_means = []
    s_means = []
    v_means = []
    
    for i in range(len(props)):
        if ((props[i].intensity_mean)[0:3]).any():
            rgb_vals = (props[i].intensity_mean)[0:3]
            hsv_vals = rgb2hsv(rgb_vals)
            h_means.append(hsv_vals[0])
            s_means.append(hsv_vals[1])
            v_means.append(hsv_vals[2])
    
    sorted_hues = np.sort(h_means)

    # Compute the first quartile (Q1)
    Q1 = np.percentile(sorted_hues, 25)
    # Compute the third quartile (Q3)
    Q3 = np.percentile(sorted_hues, 75)
    # Compute the interquartile range (IQR)
    IQR = Q3 - Q1
    
    s_sd = np.std(np.array(s_means))
    v_sd = np.std(np.array(v_means))
    
    return s_sd, v_sd, IQR

#------------------------------------------------------

#-------------Return color percentages, number of predominant colors, and number of suspicious colors-----------

#get percentage of white pixels separately because it wouldn't show up properly
#in the slic segments color average

def get_white(img, mask):
    
    img = img[:,:,:3]
    
    #segment image using mask
    img[mask == 0] = 0
    
    #convert to grayscale so we have a simple 0-1 scale for the shades
    bw = rgb2gray(img)
    
    #perfect white is 1, through testing we have found that this number seems
    #to work quite well, but this can always be changed
    WHITE_THRESHOLD = 0.65

    white_pixels = np.sum(bw > WHITE_THRESHOLD)
    
    total_pixels = np.sum(mask)
    
    percent = white_pixels/total_pixels
    
    return percent


#segment the image using the slic algorithm to get segments with similar color

def get_mean_colors(img, mask, SEG_COUNT):
    
    #get slic segments for the lesion only (mask argument makes it not take into account the background)
    segments_slic = segmentation.slic(img[:, :, :3], n_segments=SEG_COUNT, mask = mask, compactness=3, sigma=3, start_label=1)
    
    props = regionprops(segments_slic, intensity_image=img)
    color_means = []
    
    for i in range(len(props)):
        
        #get the mean color of the slic segment and convert it to rgb format
        if ((props[i].intensity_mean)[0:3]).any():
            r = round((props[i].intensity_mean[0]) * 255)
            g = round((props[i].intensity_mean[1]) * 255)
            b = round((props[i].intensity_mean[2]) * 255)
            rgb_color = (r,g,b)
            color_means.append(rgb_color)
    
    
    return color_means


#use manhattan distance to find the shade which a given rgb value is the closest to

def getColorName(R,G,B):
    
    path = "utils/mappings.csv"
    index=["color","color_name","hex","R","G","B","simple"]
    csv = pd.read_csv(path, names=index, header=None)
    #keep track of the minimum distance
    minimum = 10000
    for i in range(len(csv)):
        
        #manhattan distance
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        
        if(d<=minimum):
            minimum = d
            #this is the actual shade it is the closest to
            cname = csv.loc[i,"color_name"]
            #this is a simple mapping of the shade to a basic color, eg "blood red" to just "red"
            csimple = csv.loc[i, "simple"]
            
    return (cname, csimple)


def get_color_data(rgb_list, SEG_COUNT):
    
    #we will keep the percentage of each base color we take into account
    color_percentages = {"Red": 0, "White": 0, "Black": 0, "Blue": 0, "Yellow": 0, "Pink": 0, "Purple": 0, "Light-Brown":0,
             "Dark-Brown": 0, "Gray-Blue": 0, "Green": 0}
    
    #for the "predominant colors" feature we will only take into consideration the colors which
    #occur in at least 5% of the slic segments - again, we found this number by trial and in need this
    #can be changed
    THRESHOLD = 0.05
    
    colors = Counter()
    
    #count how many time each simple color occurs using a Python counter (each segment = one color, as we are
    #taking into consideration the mean colors of the slic segments)
    for rgb_val in rgb_list:
        
        shade, simple = getColorName(rgb_val[0], rgb_val[1], rgb_val[2])
        colors.update({simple: 1})
    
    
    colors = list(colors.items())
    
    predominant = []
    
    #through reading research papers, we have found that these colors are often associated with cancer lesions
    suspicious_colors = ["Black", "Red", "White", "Gray-Blue", "Green", "Blue"]
    
    #counter of suspicious colors
    sus_count = 0
    
    for c in colors:
        
        #if in more than 5% of the segments, add to the predominant colors
        if c[1] >= SEG_COUNT * THRESHOLD:
            predominant.append(c[0])
            if c[0] in suspicious_colors:
                sus_count += 1
        
        #percentage of the color in the slic segments
        p = round(c[1]/SEG_COUNT, 4)
        color_percentages[c[0]] = p
        
            
    n_of_colors = len(predominant)
    
    #returns predominant colors, suspicious colors, number of predominant colors, number of suspicious colors,
    #color percentage of each color      
    return suspicious_colors, n_of_colors, sus_count, color_percentages


def get_color_features(img, mask, segcount):
    
    #threshold to consider a color as predominant
    THRESHOLD = 0.05
    
    #segment using slic and get mean color of each slic segment
    means = get_mean_colors(img, mask, 150)
    #get predominant colors, number of them, number of suspicious colors, and percentage of each color
    sus_colors, predominant_count, sus_count, colors = get_color_data(means, 150)
    #using this to add percentage of white as well
    
    #since we calculated white separately, see if we have to add it to predominant and 
    #suspicious colors
    
    red = colors["Red"]
    gray_blue = colors["Gray-Blue"]
    pink = colors["Pink"]
    dark_brown = colors["Dark-Brown"]
    
    return sus_count, red, gray_blue, pink, dark_brown

#--------------------------------------------------------------------------------------------

#-----------Return all features formatted--------------

def return_features(path_mask, path_img):
    
    mask = plt.imread(path_mask)
    dim = (250, 250)
    mask= cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
    
    color_img = plt.imread(path_img)
    color_img = cv2.resize(color_img, dim, interpolation = cv2.INTER_AREA)
    
    sym = asymmetry(mask)
    comp = compactness(mask)
    
    s_sd, v_sd, iqr = get_hsv_vals(color_img, mask)
    
    sus_count,red,gray_blue,pink,dark_brown = get_color_features(color_img,mask,150)
    
    return sym, comp, iqr, s_sd, v_sd,sus_count,red,gray_blue,pink,dark_brown

#-------------------------------------------------------

#-----------Return paths of image masks and original versions----------

def return_lesion_path_data():
    
    # assign directory
    mask_directory = "../data/masks"
    image_directory = "../data/images"

    lesion_ids = []
    mask_paths = []
    image_paths = []
    # iterate over files in
    # that directory
    for filename in os.listdir(mask_directory):
        f = os.path.join(mask_directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            f = f.replace("\\","/")
            mask_paths.append(f)
            lesion_ids.append(f[:-4])
            if filename in os.listdir(os.path.join(image_directory,'imgs_part_1')):
                image_paths.append(os.path.join(image_directory, os.path.join('imgs_part_1', filename)))
            elif filename in os.listdir(os.path.join(image_directory,'imgs_part_2')):
                image_paths.append(os.path.join(image_directory, os.path.join('imgs_part_2', filename)))
            elif filename in os.listdir(os.path.join(image_directory,'imgs_part_3')):
                image_paths.append(os.path.join(image_directory, os.path.join('imgs_part_3', filename)))
            else:
                print(f"Couldn't find the following mask's original image: {filename}. Make sure it exists.")

    # for filename in os.listdir(image_directory):
    #     f = os.path.join(image_directory, filename)
    #     # checking if it is a file
    #     if os.path.isfile(f):
    #         f = f.replace("\\","/")
    #         image_paths.append(f)

    lesions = []

    if len(lesion_ids) == len(mask_paths) and len(mask_paths) == len(image_paths):
        for i in range(len(lesion_ids)):
            lesion_data = (lesion_ids[i], mask_paths[i], image_paths[i])#
            lesions.append(lesion_data)
    else:
        print("Mask and image directories contain a different number of images!")
        print(len(mask_paths))
        print(len(image_paths))
        
        print(mask_paths[-1])
        print(mask_paths[-2])
        
    return lesions

#------------------------------------------------------------------------

#---------------Create the features csv------------------

def create_features_csv():
    
    ds_path = "../data/metadata.csv"

    metadata = pd.read_csv(ds_path)
    
    lesions = return_lesion_path_data()

    header = "ID,Asymmetry,Border_Irregularity,IQR,S_STD,V_STD,N_Sus,Red" + ",Gray-Blue,Pink,Dark-Brown,Diagnostic,Is_Cancer\n"
    txt = header

    for i in range(len(lesions)):
        les = lesions[i]
        les_id = (les[0].split("/"))[-1]
        mask_ = les[1]
        img_ = les[2]
        id_ = les_id + ".png"
        
        sym, comp, iqr, s_sd, v_sd,sus_count,red,gray_blue,pink,dark_brown= return_features(mask_, img_)
        
        #get the diagnostic from the metadata csv
        use = (((metadata[metadata["img_id"] == id_])["diagnostic"]).values)[0]
        
        #depending on diagnostic update if it's cancer
        diagnostic = use
        if diagnostic in ["SCC", "BCC", "MEL"]:
            is_cancer = True
        else:
            is_cancer = False
        
        values = f"{les_id},{sym},{comp},{iqr},{s_sd},{v_sd},{sus_count},{red}," + f"{gray_blue},{pink},{dark_brown},{diagnostic},{is_cancer}\n"
        txt += values

    #print(txt)

    out_path = "features/features.csv"

    f = open(out_path, "w")
    f.write(txt)
    f.close()
    print("Done!")
    
#---------------------------------------------------------

#---------Creating the function that returns the features as an array---------------

def extract_features(img, mask):
    
    dim = (250, 250)
    mask= cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
 
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    sym = asymmetry(mask)
    comp = compactness(mask)
    
    s_sd, v_sd, iqr = get_hsv_vals(img)
    
    sus_count,red,gray_blue,pink,dark_brown = get_color_features(img,mask,150)
    x = [sym, comp, iqr, s_sd, v_sd, sus_count, red, gray_blue, pink ,dark_brown]
    return x

#-----------------------------------------------------------------------------------


#create_features_csv()
# mask = plt.imread('../data/masks/PAT_381_775_566.png')
# img = plt.imread('../data/images/imgs_part_1/PAT_381_775_566.png')
# print(extract_features(img,mask))
