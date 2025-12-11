import cv2

def preprocess(image, is_bgr=True):
    """
    Preprocess image for the model.
    
    Args:
        image: Input image
        is_bgr: If True, image is BGR (from cv2.imread). 
                If False, image is RGB (from PIL/simulator).
    """
    dimensions = image.shape
    image_w = dimensions[1]
    img_size = (200,66)

    #crop image
    x_start = 0
    y_start = 60
    width = image_w
    height = 135 - 60
    cropped_image = image[y_start : y_start + height, x_start : x_start + width]

    #convert to YUV color
    if is_bgr:
        img_yuv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2YUV)
    else:
        img_yuv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2YUV)

    #resize image
    img = cv2.resize(img_yuv, img_size)

    #apply Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    #Normalize values
    img = img/255

    return img