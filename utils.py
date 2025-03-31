import PIL.PngImagePlugin
import numpy as np
from PIL import Image
import PIL



def cal_brightness(image: PIL.PngImagePlugin.PngImageFile, average=True, option=1):
    """
    Calculate the brightness of each pixel in the image.
    :param image: PIL Image
    :return: numpy array of brightness values
    """
    # REFERENCE:
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    # https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
    
    ##########################################################################################
    
    # Formula to calculate brightness:
    # L = 0.299 * R + 0.587 * G + 0.114 * B
    # L = sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
    
    ###########################################################################################

    # the brightness can be calculated by direct tranform from RGB to L (grey scale) to evalute.
    
    
    # convert to one single channel
    gray_image = image.convert("L") # # 0.299 * R + 0.587 * G + 0.114 * B
    
    # Convert the grayscale image to a numpy array
    gray_array = np.array(gray_image)
    
    if option == 2:
        # Calculate brightness using the second formula
        gray_array = np.sqrt(gray_array ** 2 + gray_array ** 2 + gray_array ** 2)
    
    if average:
        return np.mean(gray_array)
    else:
        return gray_array



if __name__ == "__main__":
    test_high_pic = Image.open("./LOLdataset/test/high/1.png")
    test_low_pic = Image.open("./LOLdataset/test/low/1.png")
    cal_brightness(test_high_pic)
    print(f"light of the high image: {cal_brightness(test_high_pic)}")
    print(f"light of the low image: {cal_brightness(test_low_pic)}")