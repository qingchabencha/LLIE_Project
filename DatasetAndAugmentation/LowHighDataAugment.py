from torchvision import transforms
import random
import numpy as np
from PIL import Image

import albumentations as A
class PairedTransforms:
    def __init__(self, image_size=(400, 600), train=True, bright_lumination_adjust_level=-1):
        """
        Initialize the PairedTransforms class.
        if train=True: 
            This class is used to apply data augmentation and transformations to a pair of low-light and bright images 
        if bright_lumination_adjust_level > 0:
            The image with higher brightness will be adjusted to the level of brightness
    
        Args:
            image_size (tuple, optional): The size of the image, Defaults to (400, 600).
            train (bool, optional): if this pairtransform apply to training dataset, Defaults to True.
            bright_lumination_adjust_level (int, optional): if this pairtransoform need to adjust the illumination of the bright. Defaults to -1.
        """
        self.image_size = image_size
        self.train=train
        
        ##############################################################################
        
        # Define a simultaenoous transform by albumentation
        
        ##############################################################################
        # Define a simultaenoous transform by albumentation
        # https://albumentations.ai/docs/examples/example_multi_target/
        crop_width = random.randint(100,600)
        crop_height  = random.randint(100,400)
        self.augment_both = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(width=crop_width, height=crop_height, p=1),
            A.Resize(width=image_size[1], height=image_size[0], p=1),
        ], additional_targets={"imageHigh":"image"})
        
        ##############################################################################
        
        # Define the transform only apply for the bright image or the low-light image
        # Probably add different size of noise and shift
        
        ##############################################################################
        
        self.augment_bright = A.Compose([])
        self.augment_low = A.Compose([])
        
        ##############################################################################
        
        # Define the transform to convert the image to tensor and tensor to image
        
        ##############################################################################
        self.PIL_to_tensor = transforms.Compose([
            transforms.ToTensor(),  # [0,255] → [0,1]
            # The normalization can be conducted in the model to avoid the logic of reverse transformation from torch to PIL overly complex
        ])
        self.tensor_to_PIL = transforms.Compose([
            transforms.ToPILImage(),
        ])
        
        ##############################################################################
        
        # Define the transform to adjust the brightness of the bright image
        
        ##############################################################################
        self.bright_lumination_adjust_level = bright_lumination_adjust_level
        if self.bright_lumination_adjust_level > 0:
            self.bright_lumination_adjust = A.Compose([]) # TODO: implement the brightness adjustment according to the level
        

    def __call__(self, low_img, bright_img):
        # low light will be imput
        # simultaneously transform both low and bright images, if it is during training
        low_img = np.array(low_img)
        bright_img = np.array(bright_img)
        if self.train:
            transformed_image = self.augment_both(image=low_img, imageHigh=bright_img)
            low_img = transformed_image['image']
            bright_img = transformed_image['imageHigh']
            
            # apply the transform only for the bright image
            bright_img = self.augment_bright(image=bright_img)['image']
            # apply the transform only for the low-light image
            low_img = self.augment_low(image=low_img)['image']
        
            # adjust the brightness of the bright image to enable the dynamic adjustment
            if self.bright_lumination_adjust_level > 0:
                bright_img = self.bright_lumination_adjust(image=bright_img)
        
        low_img = self.PIL_to_tensor(low_img)
        bright_img = self.PIL_to_tensor(bright_img)

        return low_img, bright_img

    def tensor2PIL(self, img1=None, img2=None):
        """
        Convert the tensor to PIL image
        """
        if img1 is not None:
            img1 = self.tensor_to_PIL(img1)
        if img2 is not None:
            img2 = self.tensor_to_PIL(img2)
        return img1, img2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print('test the PairedTransforms class')
    ## Example usage
    # load the image
    img_test_bright_origin = Image.open("./LOLdataset/test/high/1.png").convert("RGB")
    img_test_low_origin = Image.open("./LOLdataset/test/low/1.png").convert("RGB")
    # create the transform class
    image_transform = PairedTransforms(image_size=(400, 600), train=True)
    # transform the image, and create the tensor
    low_img_augmented_tensor, bright_img_augmented_tensor= image_transform(low_img = img_test_low_origin, bright_img= img_test_bright_origin)
    # convert the tensor to PIL image to show the Image
    low_img_atumented_PIL, brihg_img_autmented_PIL= image_transform.tensor2PIL(low_img_augmented_tensor, bright_img_augmented_tensor)
    
    plt.subplot(2,2,1)
    plt.imshow(img_test_bright_origin)
    plt.subplot(2,2,2)
    plt.imshow(img_test_low_origin)
    plt.subplot(2,2,3)
    plt.imshow(brihg_img_autmented_PIL)
    plt.subplot(2,2,4)
    plt.imshow(low_img_atumented_PIL)
    plt.show()
    