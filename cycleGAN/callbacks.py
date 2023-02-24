import keras
import cv2
import numpy as np


class Callback():
    def __init__():
        pass

    def on_batch_end(self, models, losses):
        pass
    def on_epoch_end(self, models, losses):
        pass



class HistoryLossesCallback(Callback):
    def __init__(self, losses_names, size_limit = 1e8):
        self.cache = {}
        self.lnms = losses_names
        self.xlbl = ['Batch'] * len(losses_names)
        self.size_limit = size_limit

    def on_batch_end(self, models, batch_losses):
        for k, v in zip(self.lnms, batch_losses):
            self.append(k, v)

    def on_epoch_end(self, models, epoch_losses):
        for k, v in zip(self.lnms, epoch_losses):
            self.append(k + '-by_epoch', v)
            self.xlbl.append('Epoch')

        # if size if overlimited - shirink stat twice
        if self.get_size() >  self.size_limit:
            for k in  self.lnms:
                n = len(self.cache[k])
                self.cache[k] = self.cache[k][n//2::]
           

    def append(self, key, value):
        try:
            self.cache[key] += [value]
        except:
            self.cache[key] = [value]

    def get(self, key):
        return self.cache[key]


    def plot(self):
        import matplotlib.pyplot as plt
        c = self.cache
        xlbl = self.xlbl
        N = len(c.keys())
        fig, axes = plt.subplots(1,N, figsize=(8*N,8), dpi=100)
        for i, (k, v) in enumerate(zip(c.keys(), c.values())):
            try:
                a, b = np.polyfit(range(len(v)), v, 1)
            except:
                a, b = 0,0
            axes[i].plot(v, 'k.-', alpha=0.6,label=k)
            axes[i].plot([0, len(v)], [b, a*len(v) + b], 'r--', label='Trend')
            axes[i].set_title(k)
            axes[i].set_xlabel(self.xlbl[i])
            axes[i].legend()
        fig.tight_layout()

    def get_size(self):
        import sys
        bytes = 0.
        for k in self.cache.keys():  bytes += sys.getsizeof(self.cache[k])
        bytes += (sys.getsizeof(self.cache) + sys.getsizeof(self))
        return bytes

           



class DynamicGenOutputCallback(Callback):
    def __init__(self, datagen: keras.utils.Sequence, scale=1):
        self.scale = scale
        self.datagen = datagen


    def on_batch_end(self, models, losses):
        gen1, gen2, _, _ = models
        scale = self.scale
        x1batch, x2batch = self.datagen.__getitem__(np.random.randint(0, len(self.datagen)))
        smpl, h,w, chn = x1batch.shape
        N = min(smpl, 5)
        x1 = x1batch[0:N, ]
        x2 = x2batch[0:N, ]

        # feed geenrators
        gx2 = gen1(x1).numpy()
        gx1 = gen2(x2).numpy()

        img1 = np.zeros((h*2*scale, w*N*scale, chn)).astype('float32')
        img2 = np.zeros((h*2*scale, w*N*scale, chn)).astype('float32')

        for i in range(N):
            img1[0:h*scale, w*scale*i:w*scale*(i+1)] = np.reshape(cv2.resize(x1[i,], (h*scale, w*scale), interpolation=cv2.INTER_CUBIC), (h*scale, w*scale, chn) )
            img1[h*scale::, w*scale*i:w*scale*(i+1)] = np.reshape(cv2.resize(gx2[i,], (h*scale, w*scale), interpolation=cv2.INTER_CUBIC ), (h*scale, w*scale, chn) )
            img2[0:h*scale, w*scale*i:w*scale*(i+1)] = np.reshape(cv2.resize(x2[i,], (h*scale, w*scale), interpolation=cv2.INTER_CUBIC), (h*scale, w*scale, chn) )
            img2[h*scale::, w*scale*i:w*scale*(i+1)] = np.reshape(cv2.resize(gx1[i,], (h*scale, w*scale), interpolation=cv2.INTER_CUBIC ), (h*scale, w*scale, chn) )
            # show image

        cv2.imshow('Generator 1 output',cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        cv2.imshow('Generator 2 output',cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    # cv2.setWindowTitle('Generator', f'Generator, batch={batch}')
        cv2.waitKey(5)   
    
    def kill(self):
            cv2.destroyAllWindows()




class WeightsSaveCallback():
    def __init__(self, directory='/', frequency=1):
        import os
        self.counter = 0
        self.freq = frequency
        self.dir_ = directory + '/'
        try:
            os.mkdir(directory)
        except:
            print("WARNING: Directory already exists")
        pass

    def on_batch_end(self, models, losses):
        pass
    def on_epoch_end(self, models, losses):
        self.counter += 1
        if self.counter % self.freq: return 
        g1, g2, d1, d2 = models
        name = str(self.counter) + 'iter'
        
        g1.save_weights(self.dir_ + f'gen1_{name}.h5')
        g2.save_weights(self.dir_ + f'gen2_{name}.h5')
        d1.save_weights(self.dir_ + f'dis1_{name}.h5')
        d2.save_weights(self.dir_ + f'dis2_{name}.h5')   
        
    def load_weights(self, models, eph=None):
        g1, g2, d1, d2 = models
        if eph == None: eph = self.counter
        name = str(eph) + 'iter'
        g1.load_weights(self.dir_ + f'gen1_{name}.h5')
        g2.load_weights(self.dir_ + f'gen2_{name}.h5')
        d1.load_weights(self.dir_ + f'dis1_{name}.h5')
        d2.load_weights(self.dir_ + f'dis2_{name}.h5')
            
