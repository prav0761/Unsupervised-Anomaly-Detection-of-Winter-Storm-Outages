#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import copy
import matplotlib.colors as colors
from matplotlib import cm

class FigureGenerator:
     # plotting params
    HEAT_MAP_CELL_HEIGHT = 0.3
    SMALL_FONT = 10
    MEDIUM_FONT = 12
    LARGE_FONT = 14
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.rc('font',size=SMALL_FONT) # controls default text sizes
    plt.rc('axes',titlesize=MEDIUM_FONT) # fontsize of the axes title
    plt.rc('axes',labelsize=SMALL_FONT) # fontsize of the x and y labels
    plt.rc('xtick',labelsize=SMALL_FONT) # fontsize of the tick labels
    plt.rc('ytick',labelsize=SMALL_FONT) # fontsize of the tick labels
    plt.rc('legend',fontsize=SMALL_FONT) # legend fontsize
    plt.rc('figure',titlesize=LARGE_FONT) # fontsize of the figure title
    
    @classmethod
    def line_plot(
        cls,
        x,
        y,
        ax=None,
        color=None,
        linestyle='-',
        linewidth=1,
        fig_width=10,
        fig_height=4,
        xlabel='Date',
        ylabel='Insert label',
        xaxis_major_locator=mdates.YearLocator(1),
        xaxis_minor_locator=mdates.MonthLocator(interval=1),
        xaxis_major_fmt='%b\n%Y',
        xaxis_minor_fmt='%b\n%Y',
        xaxis_rotation=0,
        legend_loc='upper right',
        legend_labels=[],
        legend_framealpha=0,
        **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width,fig_height))
        else:
            fig = None
            pass

        ax.plot(x,y,linestyle=linestyle,color=color,linewidth=linewidth)
        ax.xaxis.set_major_locator(xaxis_major_locator)
        ax.xaxis.set_minor_locator(xaxis_minor_locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(xaxis_major_fmt))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(xaxis_minor_fmt))
        ax.tick_params('x',which='both',rotation=xaxis_rotation)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(labels=legend_labels,loc=legend_loc,framealpha=legend_framealpha)

        return fig, ax

    @classmethod
    def heat_map_plot(
        cls,
        x,
        y,
        z,
        fig=None,
        ax=None,
        diverging=True,
        cell_height=0.3,
        fig_width=10,
        shading='nearest',
        cmap_diverging='RdBu_r',
        cmap_default='Blues',
        cmap_bad='grey',
        cb_fraction=0.1,
        cb_pad=0.02,
        xlabel='Date',
        ylabel='Insert label',
        cb_label='Insert label',
        **kwargs
    ):
        if diverging:
            cmap = cmap_diverging
        else:
            cmap = cmap_default

        # heat map
        height = cell_height * len(y)
        figsize = (fig_width,height)
        divnorm = colors.TwoSlopeNorm(vcenter=0)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            pass

        # Setting colormap
        cmap = copy(cm.get_cmap(cmap))
        cmap.set_bad(color=cmap_bad)
        pcm = ax.pcolormesh(x,y,z,shading=shading,cmap=cmap,norm=divnorm,**kwargs.get('pcm_kwargs',{}))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # color bar
        cb = fig.colorbar(pcm,ax=ax,fraction=cb_fraction,pad=cb_pad,label=cb_label,**kwargs.get('cb_kwargs',{}))

        return fig, ax

    @classmethod
    def choropleth_plot(
        cls,
        df,
        column,
        ax=None,
        cmap='RdBu_r',
        edgecolor=None,
        legend=True,
        vmin=None,
        vmax=None,
        missing_kwds={'color':'grey','label':'Missing values'},
        legend_kwds={'label':'Insert label','orientation':'vertical'},
        fig_width=8,
        fig_height=8,
        **kwargs
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width,fig_height))
        else:
            fig = None
        
        if len(df[df[column].isnull()]) > 0:
            df.plot(ax=ax,column=column,cmap=cmap,edgecolor=edgecolor,legend=legend,missing_kwds=missing_kwds,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds,**kwargs)
        else:
            df.plot(ax=ax,column=column,cmap=cmap,edgecolor=edgecolor,legend=legend,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds,**kwargs)

        for spine in ['left','right','top','bottom']:
            ax.spines[spine].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        return fig, ax

