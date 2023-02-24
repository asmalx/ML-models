import numpy as np
import cv2
from scipy import ndimage


class AugmentationUnit():
    def __call__(self, image):
        return image

# nearest, mirror
class RandAugmentation(AugmentationUnit):
    def __init__(self, vshift=0., 
                 hshift=0., zoom = 0.,
                 rotate=0, vflip=False, 
                 hflip=False, normalize=True, fill_strategy='nearest'):
        
        self.vshift, self.hshift = vshift, hshift
        self.zoomf, self.rot = zoom, rotate
        self.hflip, self.vflip = hflip, vflip
        self.fill = fill_strategy
        self.norm = normalize
        self.cv2fill = cv2.BORDER_REPLICATE if fill_strategy=='nearest' else cv2.BORDER_REFLECT
        
    def __call__(self, image):
        out = image
        # shift
        vs, hs = np.random.uniform(-self.vshift/2, self.vshift/2),np.random.uniform(-self.hshift/2, self.hshift/2)
        if self.vshift or self.hshift: out = self.shift(out,vs, hs, True)
        # rotate
        if self.rot: out = self.rotate(out, np.random.randint(-self.rot,self.rot+1))
        # zoom
        if self.zoomf: out = self.zoom(out, np.random.uniform(1-self.zoomf/2, 1+self.zoomf/2))    
        # flip
        if self.hflip or self.vflip: out = self.flip(out, 0 if not self.vflip else np.random.randint(0, 2), 0 if not self.hflip else np.random.randint(0, 2))
        # norm
        if self.norm: 
            out = out - out.min()
            out = out / out.max()
            
        return out
    
    
    def rotate(self, image, angle):
        return ndimage.rotate(input=image, angle=angle, reshape=False, mode=self.fill)
    
    
    def zoom(self, image, factor):
        H, W, Ch = image.shape
        Hz, Wz = round(factor*H), round(factor*W)
        if Hz==H and Wz==W: return image
        imgz = cv2.resize(image, (Wz, Hz), cv2.INTER_CUBIC)
        if factor > 1.:
            return imgz[(Hz-H)//2:-Hz+H+(Hz-H)//2, (Wz-W)//2:-Wz+W+(Wz-W)//2,]
        else:
            return cv2.copyMakeBorder(imgz,(H-Hz)//2,H-Hz-(H-Hz)//2,(W-Wz)//2,W-Wz-(W-Wz)//2, self.cv2fill)
    
    
    def flip(self, img, vertical: bool, horizontal: bool):
        out = img.copy()
        if vertical: out = out[::-1,:, ]
        if horizontal: out = out[:, ::-1,]
        return out
    
    
    def shift(self, image, vsh, hsh, save_shape=False):
        H, W, Ch = image.shape
        Hsh, Wsh = round(vsh*H), round(hsh*W) 
        if Hsh==H and Wsh==W: return image
        a, a_ = max(Hsh, 0), max(-Hsh, 0)
        b, b_ = max(Wsh, 0), max(-Wsh, 0)
        imgsh = cv2.copyMakeBorder(image,a, a_, b, b_, self.cv2fill)

        if save_shape: 
            if Hsh==0:
                pass
            elif vsh <= 0:
                imgsh = imgsh[a_:,:,]
            else:
                imgsh = imgsh[:-a, :,]
            if Wsh==0:
                pass
            elif hsh <= 0:
                imgsh = imgsh[:, b_:,]
            else:
                imgsh = imgsh[:, :-b,]
        return imgsh