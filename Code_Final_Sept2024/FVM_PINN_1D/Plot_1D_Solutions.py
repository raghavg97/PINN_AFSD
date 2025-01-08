import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors


def save_imshow_sidebyside(plot_data):
    fixed_params =plot_data["fixed_params"]

    aspect = fixed_params['aspect']
    extent = fixed_params['extent']
    xlabel = fixed_params['xlabel']
    ylabel = fixed_params['ylabel']
    
    all_data = plot_data['data']
    n =  len(all_data)

    fig, axs = plt.subplots(1,n)
    cmap = plt.get_cmap('jet')


    
    vmax = plot_data['vmax']
    vmin = plot_data['vmin']
    colorbar_label = plot_data['colorbar_label']
    save_name = plot_data['save_name']

    norm_colors = colors.Normalize(vmax = vmax,vmin = vmin)
  
    
    for i in range(n):
         
        data = all_data[i]
        title = plot_data['title'][i]

        title_color = 'k'
        if(i==fixed_params['proposed_indicator']):
            title_color = 'r'

        ax = axs[i]
        img = ax.imshow(data,interpolation= 'none',cmap = cmap,extent=extent,aspect = aspect,vmin = vmin,vmax = vmax)
        ax.set_title(title,color =title_color,math_fontfamily = 'cm', fontsize = 16,fontweight = 'extra bold')
        ax.set_xlabel(xlabel,math_fontfamily = 'cm', fontsize = 14)
        ax.set_ylabel(ylabel,math_fontfamily = 'cm', fontsize = 14)
    
    fig.tight_layout()
    fig.colorbar(cm.ScalarMappable(norm=norm_colors,cmap = cmap),
                 ax = axs,location = 'bottom',orientation = 'horizontal',
                 label = colorbar_label,aspect = 40)


    formats = ['svg','png','eps','pdf']
    folder_base = './Plots/'
    # folders = ['svg/','png/','eps/','pdf/']

    for format in formats:
        fig.savefig(folder_base + format + '/' + save_name+'.'+format,
                    format = format,bbox_inches = 'tight')
    