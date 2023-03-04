from PIL import Image, ImageFont, ImageDraw 
import numpy as np

class DrawDTree():
    def __init__(self, root):
        # width and height of node and distance between nodes on the picture
        self.nw, self.nh = 150, 120
        self.nsw, self.nsh = 100, 50
        
        self.depth = self.__get_depth_rec(root)
        # parse tree to array of nodes arr[depth][nodes]; maxwidth        
        self.array, self.twidth = self.__parse_tree(root, self.depth)
        # calc recolution
        self.resolution = (self.nh * self.depth, self.twidth * self.nw)

        # create image array, PIL image, PIL draw obj
        self.img = (255*np.ones((*self.resolution, 3))).astype("uint8") 
        self.imgPIL = Image.fromarray(self.img)
        self.draw = ImageDraw.Draw(self.imgPIL)
        
        # draw tree
        self.__draw_tree_iterative()

    def get_image(self):
        # return array
        return np.array(self.imgPIL)

    
    def save_jpg(self, name=None):
        # save as jpg
        if name is None: name = 'tree' + str(id(self))
        self.imgPIL.save(name + '.jpg' )
    
    
    def __draw_tree_iterative(self):
        # cacl init pos of root
        mx, my =  self.resolution[1] / 2, 10
        # dicts for connection drawing
        prev_layer_pos = {0:(mx, my)}
        layer_pos = {}
        
        # iterate by depth
        for layer in self.array:
            n = len(layer) # count of nodes
            # iterate by nodes of current depth level
            for i, (node, branch, parent_id) in enumerate(layer):
                # calc positions of current node, while key is index of node
                layer_pos[i] = tuple((mx - self.nw*n/2. + self.nw/2 + i*self.nw, my))
                # draw connection with parent node
                self.__draw_connection(*prev_layer_pos[parent_id], *layer_pos[i], branch)
                # draw node
                self.__draw_node(node, *layer_pos[i])
            # mode position down, to next level
            my += self.nh
            # save indexes/nodes positions for next iteration (as parents)
            prev_layer_pos = dict(layer_pos)  
            
            
    def __draw_node(self, node, posx, posy):
        # parse node:
        # different color for branch and leaf
        if node.isLeaf: cfill = (100, 180, 255)
        else: cfill = (100, 255, 100)
     
        text = self.__parse_node(node)
        # draw node square
        posx = posx - self.nsw / 2
        self.draw.rectangle([posx, posy, posx+self.nsw, posy+self.nsh],
                            fill=cfill, outline=(0,0,0), width=1)
        
        # text pos
        text_width= 6.5*np.max([len(s) for s in text.split('\n')])
        text_heigh = 12*len(text.split('\n'))
        # draw text
        self.draw.text((posx + (self.nsw-text_width)/2, posy +(self.nsh-text_heigh)/2), text, align ="center", fill=(50,50,50) , spacing=0.25) 

    
    def __parse_node(self, node):
        # parse node data
        text = ''
        if node.isLeaf:
            text += "Samples = " + str(len(node.y)) + '\n'
            text += "lbl: [" + ','.join([str(np.sum([node.y==i])) for i in [0,1]]) + ']\n' 
            text += "stop: " + node.msg       
        else:
            text += str(node.cond) + '\n'
            text += "IG = " + str(node.IG)[0:5] + '\n'
            text += "Samples = " + str(len(node.y)) 
        return text
        
    def __get_depth_rec(self, root, depth=0):        
        # calculate tree depth recursively
        if root is None: return depth
        return max(self.__get_depth_rec(root.L, root.depth), self.__get_depth_rec(root.R, root.depth))


    def __parse_tree(self, root, depth):
        # convert tree to array[depth][width] = 
        # = [[(node, branch_index, parent_index), ...], ...]        
        max_width = 0 # max width (tree width)
        tree = [[(root, None, 0)]] # output array
        stack = [(root, None, 0)] # stack for current layer
        # fill array with all children in current depth layer
        for d in range(depth):
            children_list = []
            for idx, (node, branch, parent_idx) in enumerate(stack):
                if not node.L is None: children_list.append((node.L, -1, idx))
                if not node.R is None: children_list.append((node.R, 1, idx)) 

            tree.append(children_list)
            if max_width < len(children_list): max_width = len(children_list)
            stack = children_list

        return tree, max_width


    def __draw_connection(self, posx_from, posy_from, posx_to, posy_to, branch=None):
        # draws connection between nodes
        pyf = posy_from + self.nsh # start pos y of connection
        self.draw.line([posx_from, pyf, posx_to, posy_to], fill=(0,0,0), width=2)
        # if branch specified, draw 'true', 'false' mark
        if branch is None:
            return 
        elif branch == -1:
            self.draw.text([posx_from / 2 + posx_to / 2, pyf / 2 + posy_to / 2], 'true', align ="left", fill=(50,50,50) , spacing=0.25)
        else:
            self.draw.text([posx_from / 2 + posx_to / 2, pyf / 2 + posy_to / 2], 'false', align ="left", fill=(50,50,50) , spacing=0.25)    
