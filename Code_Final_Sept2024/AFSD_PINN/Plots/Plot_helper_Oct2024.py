import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import matplotlib.colors

import numpy as np

#Plot 3D
class plot3D_comp2():
    def __init__(self,plot_params,data_params,data):
        self.fig = plt.figure()
        self.fig.tight_layout()

        self.plot_titles = plot_params['plot_titles']
        self.colormap_name = plot_params['colormap_name']
        self.colorbar_label = plot_params['colorbar_label']
        self.colormap = plt.get_cmap(self.colormap_name)
        self.folder_base = plot_params['folder_base']
        self.save_name = plot_params['save_name']
        
        self.resol_x = data_params['resol_x']
        self.resol_y = data_params['resol_y']
        self.resol_z = data_params['resol_z']
        self.proposed_indicator = data_params['proposed_indicator']

        self.data = data
            

        if(plot_params['vmin_max'] == None):#if None infer from Left side Data
            vmin = np.min(data['left'])
            vmax = np.max(data['left'])
            self.vmin_max = {"vmin": vmin,"vmax":vmax}
            print("Vmin:",vmin)
            print("Vmax:",vmax)
        else:
            self.vmin_max = plot_params['vmin_max']


        [self.x_min,self.y_min,self.z_min] = data_params['lb_xyz']# THIS  IS FOR THE PLOT and NOT THE PROBLEM
        [self.x_max,self.y_max,self.z_max] = data_params['ub_xyz']

        x = np.linspace(self.x_min,self.x_max,self.resol_x+1)
        y = np.linspace(self.y_min,self.y_max,self.resol_y+1)
        z = np.linspace(self.z_min,self.z_max,self.resol_z+1)

        X,Y,Z = np.meshgrid(y,x,z)

        self.plot_3D_left(self.data['left'],X,Y,Z)
        self.plot_3D_right(self.data['right'],X,Y,Z)
        self.save_plots()

        print("All plots successfully saved...!")
          # This gives RGBA colors

    def plot_3D_left(self,data,X,Y,Z):
        ax = self.fig.add_subplot(121,projection='3d')

        plot_title = self.plot_titles['left']

        proposed_indicator = (self.proposed_indicator == 'l')
        self.plot_commons(data,ax,plot_title,X,Y,Z,proposed_indicator)


    def plot_3D_right(self,data,X,Y,Z):
        ax = self.fig.add_subplot(122,projection='3d')

        plot_title = self.plot_titles['right']

        proposed_indicator = (self.proposed_indicator == 'r')
        self.plot_commons(data,ax,plot_title,X,Y,Z,proposed_indicator)

        norm = matplotlib.colors.Normalize(vmin=self.vmin_max['vmin'], vmax = self.vmin_max['vmax'])
        m = cm.ScalarMappable(cmap=self.colormap, norm=norm)
        m.set_array([])
        cbaxes = self.fig.add_axes([0.35,0.25, 0.35, 0.02])
        cbar = self.fig.colorbar(m,cax = cbaxes,orientation = 'horizontal', aspect = 1)
        cbar.ax.tick_params(labelsize=12,labelbottom = True,labeltop = False, bottom =True, top = False)
        cbar.ax.set_xlabel(self.colorbar_label)



    def plot_commons(self,plot_data,ax,plot_title,X,Y,Z,proposed_indicator): #Common between left and right
        vmin = self.vmin_max['vmin']
        vmax = self.vmin_max['vmax']
       

        normalized_data = (plot_data - vmin) / (vmax - vmin)

        
        colors = self.colormap(normalized_data)
        
      
        ax.voxels(Y,X,Z,plot_data,facecolors=colors,edgecolors=colors)
        
        ax.xaxis.pane.fill = False  # Hide the background of the x-axis
        ax.yaxis.pane.fill = False  # Hide the background of the y-axis
        ax.zaxis.pane.fill = False  # Hide the background of the z-axis
        ax.xaxis.set_ticks(np.linspace(self.x_min,self.x_max,3))       # Hide tick marks for the x-axis
        ax.yaxis.set_ticks(np.linspace(self.y_min,self.y_max,3))       # Hide tick marks for the y-axis
        ax.zaxis.set_ticks(np.linspace(self.z_min,self.z_max,3))       # Hide tick marks for the z-axis
        ax.tick_params(axis='both',which='major',pad = -5, labelsize = 4)

        ax.grid(False)

        ax.xaxis.line.set_color((1, 1, 1, 0))  # Set x-axis line color to transparent
        ax.yaxis.line.set_color((1, 1, 1, 0))  # Set y-axis line color to transparent
        ax.zaxis.line.set_color((1, 1, 1, 0))  # Set z-axis line color to transparent

        # Hide grid structure

        ax.set_box_aspect([7.0, 2.5, 1.0])
        
        if(proposed_indicator == True):
            ax.set_title(plot_title,color = 'r',math_fontfamily = 'cm', fontsize = 18,fontweight = 'extra bold')
        else:
            ax.set_title(plot_title,color = 'k',math_fontfamily = 'cm', fontsize = 18,fontweight = 'extra bold')

        #Colorbar added only on the right

    def save_plots(self):
        formats = ['svg','png','eps','pdf']
        # folder_base = './Plots/'
        # folders = ['svg/','png/','eps/','pdf/']

        for format in formats:
            self.fig.savefig(self.folder_base + format + '/' + self.save_name+'.'+format,
                             format = format,bbox_inches = 'tight')
    
    