import time

import matplotlib.pyplot as plt
plt.style.use('bmh')

import callbacks

import numpy as np

'''
Class ProgressBar draws flexible progress line (like tqdm)

 - iter_started method captures the time 
 - new_line method jumps to next string (use to save previous line)
 - __call__ or progress method draws progress bar:
    iters_name - lsit [name1, name2, ...] (names of iters)
    iters_max - list [A, B, ...]
    iters_cur - list [a, b, ...]
    progress - list [X, x]
    stat - dict {'param1':<float>, ...}
 
    OUT:                            x           X-x             
        name1 a/A - name2 b/B - [===========........] - Time: ??? - param1=<value>

'''

class ProgressBar():
    def __init__(self):
        self.iter_started()
        self.prev_len= 10
        
        
    def iter_started(self):
        self.start_time = time.time()
        
        
    def new_line(self):
        self.print_out('\n ')
        
    
    def __call__(self, iters_name, iters_max, iters_cur, progress=[1,1], stat={}):
        self.progress(iters_name, iters_max, iters_cur, progress, stat)
    
    
    def progress(self, iters_name, iters_max, iters_cur, progress=[1,1], stat={}):
        out = ''
        time_stop = time.time()
        if len(iters_name) > 0:
            self.itnames = iters_name
        if len(iters_max) > 0:
            self.iters_max = iters_max
        
        try:
            out = ' - '.join([f'{iters_name[i]} {k}/{iters_max[i]}' for i, k in enumerate(iters_cur)])
        except:
            out = '<iterations invalid input> '
                    
        if len(progress) > 1:
            frac = progress[1] / progress[0]
            a,b = int(20*frac), 20-int(20*frac)
            out = out + ' - |' + '█'*a + '.'*b + '| - '               

            
        time_s = time_stop - self.start_time
        if time_s != 0:
            t = [-1,-1,-1]
            d = ''
            time_ = round(time_s)
            if time_ // 86400 > 0:  d = str(time_ // 86400) + 'd '
            if time_ // 3600 > 0:   t[2] = (time_ // 3600) % 24
            if time_ // 60 > 0:     t[1] = (time_ // 60) % 60
            t[0] = time_ % 60
            out = out + "Time: "+ d+ ':'.join(['0'*(2-len(str(i))) + str(i) for i in t[::-1] if i >=0] )

            for k, v in zip(stat.keys(), stat.values()):
                out =out + ' - ' + str(k)+'='+self.__round_and_str(v)
            out = out[:-2]
            
            # printing
        self.print_out(out)
  

    def print_out(self, out):
        print(out +' '*max(self.prev_len-len(out), 1), end='\r')
        self.prev_len = len(out)  
        


    def __round_and_str(self, value):
        out = round(value, 4)
        if 0 < value < 0.0001:
            out = self.format_e(value, prec=3)
        if value > 99:
            out = round(value, 2)
        if value > 999:
            out = round(value, 1)
        if value > 9999:
            out = round(value)
        if value > 999999:
            out = self.format_e(value, prec=2)
        if value < 0.1**5:
            out = self.format_e(value, prec=2)
        return str(out)

    
    def format_e(self, n, prec=5):
        a = '%E' % n
        x, ys = round(float(a.split('E')[0].rstrip('0').rstrip('.')),prec) , a.split('E')[1]
        return str(x) + 'E' + ys


###########################################
#class Gen

def plot_history(history):
    fig, axes = plt.subplots(1,2, figsize=(16,8), dpi=100)
    axes[0].plot(history['epoch'],history['generators_loss'], label='Generator loss')
    axes[1].plot(history['epoch'], history['discriminators_loss'], label='Discriminators loss')
    for ax in axes: 
        ax.legend()
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    plt.show(block=True)
    return fig








def pack_into_array(x, shape, map_func=lambda x: x):
    w, h = x.shape[1:]
    m ,n = shape
    arr = np.zeros((w*m, h*n))
    for j in range(m): 
        for i in range(n): 
            arr[j*w:(j+1)*w, i*h:(i+1)*h,] = map_func(x[i+j*n])
    return arr

def pack_into_array3dim(x, shape, map_func=lambda x: np.clip(x, 0., 1.)):
    w, h, ch = x.shape[1:]
    m ,n = shape
    arr = np.ones((w*m, h*n, ch))
    for j in range(m): 
        for i in range(n): 
            arr[j*w:(j+1)*w, i*h:(i+1)*h, :] = map_func(x[i+j*n])
    return arr


import cv2


