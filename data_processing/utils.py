from ast import Return
import re
from PIL import Image
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
import torchvision.transforms as transforms

import torchvision.transforms.functional as F
import torch
from skimage import filters, measure
import imageio as iio
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn import preprocessing
	
import matplotlib.patches as mpatches
import cv2

def detect_xyr(img):
    """
    Taken from https://github.com/linchundan88/Fundus-image-preprocessing/blob/master/fundus_preprocessing.py
    """
    
    # determine the minimum and maximum possible radii
    MIN_REDIUS_RATIO = 0. #cambie
    MAX_REDIUS_RATIO = 1.0
    # get width and height of the image
    width = img.shape[1]
    height = img.shape[0]

    # get the min length of the image
    myMinWidthHeight = min(width, height)
    # and get min and max radii according to the proportion
    myMinRadius = round(myMinWidthHeight * MIN_REDIUS_RATIO)
    myMaxRadius = round(myMinWidthHeight * MAX_REDIUS_RATIO)

    # turn the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # estimate the circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=200, param2=0.9,
                               minRadius=myMinRadius,
                               maxRadius=myMaxRadius)

    (x, y, r) = (0, 0, 0)
    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            x1, y1, r1 = circles[0]
            if x1 > (2 / 5 * width) and x1 < (3 / 5 * width) \
                    and y1 > (2 / 5 * height) and y1 < (3 / 5 * height):
                x, y, r = circles[0]
                found_circle = True


    if not found_circle:
        # sum the R, G, B channels to form a single image
        sum_of_channels = np.asarray(np.sum(img,axis=2), dtype=np.uint8)
        # threshold the image using Otsu
        fov_mask = sum_of_channels > filters.threshold_otsu(sum_of_channels)
        # fill holes in the approximate FOV mask
        fov_mask = np.asarray(binary_fill_holes(fov_mask), dtype=np.uint8)

        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]
        side_2 = coordinates[3] - coordinates[1]
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)

    return found_circle, x, y, r

def get_fov_mask(fundus_picture):
    '''
    Obtains the fov mask of the given fundus picture.
    '''

    fov_mask = np.zeros((fundus_picture.shape[0], fundus_picture.shape[1]), dtype=np.uint8)
    
    # estimate the center (x,y) and the radius (r) of the circle
    _, x, y, r = detect_xyr(fundus_picture)
    

    Y, X = np.ogrid[:fundus_picture.shape[0], :fundus_picture.shape[1]]
    dist_from_center = np.sqrt((X - x)**2 + (Y-y)**2)

    fov_mask[dist_from_center <= r] = 255
                
    return fov_mask, (x,y,r)

def crop_fov(fundus_picture, mask, fov_mask= None):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    # if the FOV mask is not given, estimate it
    if fov_mask is None:
        fov_mask, (x, y, r) = get_fov_mask(fundus_picture)
    # if the FOV mask is given, estimate the center and the radii
    else:
        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]
        side_2 = coordinates[3] - coordinates[1]
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)

    lim_x_inf = (y - r) if y>=r else 0
    lim_x_sup = (y + r) if (y+r)<fundus_picture.shape[0] else fundus_picture.shape[0]
    lim_y_inf = (x - r) if x>=r else 0
    lim_y_sup = (x + r) if (x+r)<fundus_picture.shape[1] else fundus_picture.shape[1]

    return (fundus_picture[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup,:], 
        mask[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup])

def crop_fov_2(fundus_picture,mask1, mask2, fov_mask = None):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    # if the FOV mask is not given, estimate it
    if fov_mask is None:
        fov_mask, (x, y, r) = get_fov_mask(fundus_picture)
    # if the FOV mask is given, estimate the center and the radii
    else:
        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]
        side_2 = coordinates[3] - coordinates[1]
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)

    lim_x_inf = (y - r) if y>=r else 0
    lim_x_sup = (y + r) if (y+r)<fundus_picture.shape[0] else fundus_picture.shape[0]
    lim_y_inf = (x - r) if x>=r else 0
    lim_y_sup = (x + r) if (x+r)<fundus_picture.shape[1] else fundus_picture.shape[1]

    return (fundus_picture[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup,:], 
        mask1[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup],
        mask2[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup])

def crop_fov_limits(fundus_picture, fov_mask= None):
    '''
    Extract an approximate FOV mask, and crop the picture around it,
    return the limits of the image to cut
    '''

    # if the FOV mask is not given, estimate it
    if fov_mask is None:
        fov_mask, (x, y, r) = get_fov_mask(fundus_picture)
    # if the FOV mask is given, estimate the center and the radii
    else:
        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox#(min_row, min_col, max_row, max_col)
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]#row
        side_2 = coordinates[3] - coordinates[1]#col
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)

    lim_x_inf = (y - r) if y>=r else 0
    lim_x_sup = (y + r) if (y+r)<fundus_picture.shape[0] else fundus_picture.shape[0]
    lim_y_inf = (x - r) if x>=r else 0
    lim_y_sup = (x + r) if (x+r)<fundus_picture.shape[1] else fundus_picture.shape[1]

    return lim_x_inf,lim_x_sup, lim_y_inf,lim_y_sup, r


def rescale_test(fundus_img, mask, target_radio=255):
    '''
    Para los datos de test buscamo reducir la imagen de manera tal que el radio del disco en la imagen de test se asimile al tamano del disco en la imagen de train
    fundu
    inputs: -----------------------------
    fundus_img: la imagen de entrada
    mask: mascara correspondiente a la imagen dada
    target_radio: tamano en el que esperamos que este el radio del disco de la imagen aproximadamente
    '''
    #print('ENTRO: ')
    lim_x_inf,lim_x_sup, lim_y_inf,lim_y_sup, radio = crop_fov_limits(fundus_img)
        
    
    width, height,_ = fundus_img.shape    
    scale = target_radio / radio

    new_width = int(scale*width)
    new_height = int(scale*height)

    scaledImg = resize(fundus_img, (new_width,new_height))
    
    final_size = max(new_width,new_height)
    square_Img = resize(scaledImg,(final_size,final_size))
    if mask != []:
        y_rescale = resize(mask, (new_width,new_height))
        square_mask = resize(y_rescale,(final_size,final_size))
    else:
        square_mask = None
    
        
    return square_Img,square_mask

def get_OCOD_crop(mask,delta, descentro= False):
    ''' dada la mascara de una imagen, calculo la zona del disco y la copa + un delta
    se toma la zona clasificada, de ella al lado mas largo (alto u ancho) se le agrega el factor alpha, 
    el lado restante se lo acomoda para que el resultado final sea una imagen cuadrada

    update: obtener una desviacion del centro random que genere imagenes que no posean el disco en el entro
    ------------------------------------
    input
    mask: imagen binaria de la clasificacion
    delta: indice en porcentaje de que tanto mas de la mascara queremos dejan en los bordes
    descentro: booleano que nos indica si queresmos obicar el centro en otra parte de la imagen de manera aleatoria
    ------------------------------------
    output
    cordenadas para el recorte de la imagen
    centro de la imagen de recorte 
    '''

    #mask = mask[0,:,:]
    #print('VALUES OF THE OUTPUT WITH ARGMAX: ', np.unique(mask))
    thresh = filters.threshold_otsu(mask) #USAMOS OTSU PARA LA GENERACION DE IMAGENES
    binary = mask > thresh
    
    blobs_labels = measure.label(binary, background=0)
    
    region = measure.regionprops(blobs_labels)

    

    biggest_region = -1
    biggest_area = -1
    for r in region:
        #print(r.area)
        if biggest_area < r.area:
            biggest_area = r.area
            biggest_region = r
        #print('biggest area: ', biggest_region)
    if biggest_area == -1:
       
        print('NO CLASIFICATION ON THE FIRST STEP')
        #UTILIZAR OTSU PARA SEGMENTACION BINARIA 
        
        plt.imshow(binary)
        #plt.imshow(th2)

        plt.show()
        #return 0, 0, 512, 512, (255,255)


    
    coordinates = biggest_region.bbox #(min_row, min_col, max_row, max_col)
    #print(coordinates)
    #coordinates = region[0].bbox #(min_row, min_col, max_row, max_col)
    

    coord0 = coordinates[0]
    coord1 = coordinates[1]
    coord2 = coordinates[2]
    coord3 = coordinates[3]

    # rect=mpatches.Rectangle((coord1,coord0), coord3 - coord1,coord2 - coord0, 
    #                     fill = False,
    #                     color = "purple",
    #                     linewidth = 2)
    # plt.imshow(blobs_labels)
    # plt.gca().add_patch(rect)
    # plt.show()

    sub_w = coord3 - coord1 #que al revez para que sea positivo
    if sub_w < 0 :
        sub_w = sub_w + 2 * sub_w
    sub_h = coord2 - coord0 
    #print('SUB H: ', sub_h, 'SUB_W: ', sub_w)


    if descentro == True:
        max_h = int(sub_h * delta)
        max_w = int(sub_w * delta)
        #print('MAX H: ', max_h, 'MAX_W: ', max_w)
        rand_h = torch.randint(low=0,high=max_h+1,size=(1,))
        rand_w = torch.randint(low=0,high=max_w+1,size=(1,))
        rest_h = torch.rand(1)
        if rest_h > 0.5:
            coord0 -= rand_h
            coord2 -= rand_h
        else:
            coord0 += rand_h
            coord2 += rand_h

        rest_w = torch.rand(1)
        if rest_w > 0.5:
            coord1 -= rand_w
            coord3 -= rand_w
        else:
            coord1 += rand_w
            coord3 += rand_w


    x = coord1 + (sub_w // 2)
    y = coord0 + (sub_h // 2)
    if sub_w > sub_h: #REVISAR LOS LIMITES
        #print('ENTRO')

        c1 = int(coord1 - coord1 * delta)
        c3 = int(coord3 + coord3 * delta)
        new_h = c3 - c1 
        if (new_h % 2) != 0:
            new_h = new_h + 1
            c3 = c3 + 1
        new_w = new_h
        half = new_w / 2
        c0 = y - half
        c2 = y + half
    else:
        c0 = int(coord0 - coord0 * delta)
        c2 = int(coord2 + coord2 * delta)
        new_w = c2 - c0 
        if (new_w % 2) != 0:
            new_w = new_w + 1
            c2 = c2 + 1
        new_h = new_w
        half = new_w / 2
        c1 = x - half
        c3 = x + half
    #print( 'Mask shape in train ', mask.shape)
    h,w  = mask.shape

    #VERIFICO LIMITES

    if c2 > h:
        c2 = h
    if c3 > w:
        c3 = w
    if c0 < 0:
        c0 = 0
    if c1 < 0:
        c1 = 0
    #print(int(c0),int(c1),int(c2),int(c3))
    return int(c0),int(c1),int(c2),int(c3), (x,y), binary

def stage1_preprocessing(img, norm = True):
    tr = transforms.ToTensor()
    #img.cpu()
    if norm:
        img = normalization(np.array(img[0,:,:,:],dtype=float))
    
    img = resize(img, (512, 512),preserve_range=True)
            
    tensor_img = tr(img) #convierto en tensor
    tensor_img = torch.unsqueeze(tensor_img, 0) #adding the batch dimension
    return tensor_img

def normalization(img):
    #NORMALIZO LA IMAGEN, 
    #Obtengo media y desvio por cada imagen y normalizo la imagen
    #posteriormente la resizeo a un valor mas chico de 512S,512

    #img = img.float()
    #shape = img.size() #Guardo los tamaños originales

    mean_r = np.mean(img[:,:,0])
    mean_g = np.mean(img[:,:,1])
    mean_b = np.mean(img[:,:,2])

    std_r = np.std(img[:,:,0])+1e-6
    std_g = np.std(img[:,:,1])+1e-6
    std_b = np.std(img[:,:,2])+1e-6

    norm_img = np.zeros((img.shape))
    norm_img[:,:,0] = (img[:,:,0]-mean_r) / std_r
    norm_img[:,:,1] = (img[:,:,1]-mean_g) / std_g
    norm_img[:,:,2] = (img[:,:,2]-mean_b) / std_b

    return norm_img

def stage2_preprocessing(img,ODpred, delta):
    tr = transforms.ToTensor()
    b, h, w, c= img.shape #[batch, height, weight, channel]

    

    out = ODpred[0,1,:,:] #me quedo con la probabilidad de que sea disco [1,2,512,512]
    out = np.array(out) #[ 512, 512]
    #print('SHAPE: ', out.shape)
    
    
    out = resize(out,(h,w),preserve_range=True)#[h,w]
    #print('SHAPE OUT: ', out.shape)
    
    arg = out>=0.5
    #print('SHAPES OF THE  MASK: ', out.shape)
    
    #print('SHAPES OF THE BINARY MASK: ', arg.shape)

    
    c0, c1, c2, c3, centro, b_out = get_OCOD_crop(mask= out, delta= delta, descentro= False)
    
    # plt.imshow(out[c0:c2,c1:c3])
    # plt.show()
    
    img = normalization(np.array(img[0,:,:,:],dtype=float)) #[h,w,c]
    cut_img = img[c0:c2,c1:c3,:]
    #cut_img = resize(cut_img,(512,512),preserve_range=True) #VUELVO LA IMAGEN A 512X512
    cut_img = tr(cut_img) #LA DEVUELVO COMO TENSOR
    cut_img = torch.unsqueeze(cut_img, 0) #adding the batch dimension
    return c0,c1,c2,c3, cut_img, b_out   




# imagen = '/mnt/Almacenamiento/ODOC_segmentation/data/images/IDRID/001.png'
# mask = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/IDRID/001.png'

# img = iio.imread(imagen)
# mk = iio.imread(mask)

# new_img, new_mask = crop_fov(img,mk)


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(new_img)
# ax2.imshow(new_mask)

# plt.show()



