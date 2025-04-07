
import copy
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as mcol
from matplotlib.colors import LogNorm, Normalize
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

currentFig = 1
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['text.usetex'] = True

def display_image_simple_with_contour(image, contour_image, levels, bad='black',
                                      cbar_label='', cmap=cm.viridis, vmin=None,
                                      vmax=None, figsizewidth=9, figsizeheight=9) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    cmap.set_bad(bad, 1)
    
    norm = LogNorm(vmin=vmin, vmax=vmax)
    frame = ax.imshow(image, origin='lower', cmap=cmap, norm=norm)
    cbar = plt.colorbar(frame)
    cbar.set_label(cbar_label, fontsize=15)
    
    ax.contour(contour_image, levels=levels, colors='white', alpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    return

def display_image_simple(data, bad='black', cbar_label='', cmap=cm.inferno, 
                         vmin=None, vmax=None, figsizewidth=9, figsizeheight=9,
                         lognorm=True, save=False, outfile=None, title=None,
                         patches=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap) # cmap=cm.inferno # cmap=cm.viridis,
    cmap.set_bad(bad, 1)
    
    if lognorm :
        norm = LogNorm(vmin=vmin, vmax=vmax)
        frame = ax.imshow(data, origin='lower', cmap=cmap, norm=norm)
    else :
        frame = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(frame)
    cbar.set_label(cbar_label, fontsize=15)
    
    if patches is not None :
        for patch in patches :
            patch.plot(ax=ax, color='white', lw=0.5)
    
    ax.set_title(title, fontsize=18)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def histogram(data, label, title=None, bins=None, log=False, histtype='step',
              vlines=[], colors=[], labels=[], loc='upper left',
              figsizewidth=9.5, figsizeheight=7, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    # if bins and not log :
    ax.hist(data, bins=bins, color='k', histtype=histtype, log=log)
    # elif bins and log :
    #     ax.hist(data, bins=bins, log=log, color='k', histtype=histtype)
    # elif log and not bins :
    #     ax.hist(data, log=log, color='k', histtype=histtype)
    # else :
    #     ax.hist(data, color='k', histtype=histtype)
    
    # if len(vlines) > 0 :
    #     for i in range(len(vlines)) :
    #         ax.axvline(vlines[i], ls='--', color=colors[i], lw=1, alpha=0.5,
    #                    label=labels[i])
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('{}'.format(label), fontsize = 15)    
    
    # if len(vlines) > 0 :
    #     ax.legend(loc=loc, facecolor='whitesmoke', framealpha=1,
    #               fontsize=15)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def histogram_multi(data, hlabel, colors, styles, labels, bins, #weights,
                    xmin=None, xmax=None, ymin=None, ymax=None, title=None,
                    figsizewidth=9.5, figsizeheight=7, loc=0, histtype='step') :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for i in range(len(data)) :
        ax.hist(data[i], color=colors[i], linestyle=styles[i], label=labels[i],
                histtype=histtype, bins=bins[i]) # weights=weights[i]
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(hlabel, fontsize=15)
    # ax.set_ylabel('Fractional Frequency', fontsize=15)
    
    # ax.set_xscale('log')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc=loc, facecolor='whitesmoke', framealpha=1,
              fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    return

def histogram_2d(xhist, yhist, label=None, #xscatter, yscatter, xs, ys, fitx, fity, labels,
                 # styles,
                 bad='white', bins=[20,20], cmap=cm.Blues, title=None,
                 norm=LogNorm(), outfile=None, xlabel=None, ylabel=None,
                 xmin=None, xmax=None, ymin=None, ymax=None, save=False,
                 figsizewidth=9.5, figsizeheight=7, loc=0) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    cmap.set_bad(bad, 1)
    
    ax.hist2d(xhist, yhist, bins=bins, #range=[[xmin, xmax], [ymin, ymax]],
               cmap=cmap, norm=norm, alpha=0.7, label=label)
    
    # for i in range(len(ys)) :
    #     ax.plot(xs[i], ys[i], styles[i], color='k')
    
    # for i in range(len(fity)) :
    #     ax.plot(fitx[i], fity[i], 'r-', label=labels[i])
    
    # ax.plot(xscatter, yscatter, 'ro', label=labels[-1])
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.legend(loc=loc, facecolor='whitesmoke', framealpha=1, fontsize=15)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_CASTOR_proposal(title1, title2, title3, df1, df2, df3,
                         hist1, hist2, hist3, fwhm1, fwhm2, fwhm3,
                         xs, main1, main2, main3,
                         lo1, lo2, lo3, med1, med2, med3, hi1, hi2, hi3, XX, YY,
                         xlabel_t=None, ylabel_t=None, xlabel_b=None, ylabel_b=None,
                         xmin_t=None, xmax_t=None, ymin_t=None, ymax_t=None,
                         xmin_b=None, xmax_b=None, ymin_b=None, ymax_b=None,
                         save=False, outfile=None, label=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(13, 9))
    currentFig += 1
    plt.clf()
    
    gs = fig.add_gridspec(2, 3, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    norm=LogNorm(vmin=1e-11, vmax=2e-09)
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax1.pcolormesh(XX, YY, hist1, cmap=cmap, norm=norm, alpha=0.9)
    sns.kdeplot(data=df1, x='dx', y='dz', color='lime', weights='masses',
                levels=[0.1, 0.2, 0.3, 0.6, 0.9], ax=ax1, linewidths=3)
    circle1_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle1_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax1.add_patch(circle1_in)
    ax1.add_patch(circle1_out)
    beam1 = Circle((4, -4), radius=fwhm1, facecolor='none', ls='-',
                        edgecolor='w', linewidth=1.5, alpha=1, zorder=3)
    ax1.add_patch(beam1)
    ax1.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax1.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax1.set_title(title1, fontsize=15)
    ax1.set_xlabel(xlabel_t, fontsize=15)
    ax1.set_ylabel(ylabel_t, fontsize=15)
    ax1.set_xlim(xmin_t, xmax_t)
    ax1.set_ylim(ymin_t, ymax_t)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    # ax1.axes.set_aspect('equal')
    
    ax2.pcolormesh(XX, YY, hist2, cmap=cmap, norm=norm, alpha=0.9)
    sns.kdeplot(data=df2, x='dx', y='dz', color='lime', weights='masses',
                levels=[0.1, 0.2, 0.3, 0.6, 0.9], ax=ax2, linewidths=3)
    circle2_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle2_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax2.add_patch(circle2_in)
    ax2.add_patch(circle2_out)
    beam2 = Circle((4, -4), radius=fwhm2, facecolor='none', ls='-',
                        edgecolor='w', linewidth=1.5, alpha=1, zorder=3)
    ax2.add_patch(beam2)
    ax2.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax2.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax2.set_title(title2, fontsize=15)
    ax2.set_xlabel(xlabel_t, fontsize=15)
    ax2.set(ylabel=None)
    ax2.set_xlim(xmin_t, xmax_t)
    ax2.set_ylim(ymin_t, ymax_t)
    ax2.tick_params(axis='x', which='major', labelsize=11)
    ax2.yaxis.set_ticks([])
    ax2.yaxis.set_ticklabels([])
    # ax2.axes.set_aspect('equal')
    
    image3 = ax3.pcolormesh(XX, YY, hist3, cmap=cmap, norm=norm, alpha=0.9)
    sns.kdeplot(data=df3, x='dx', y='dz', color='lime', weights='masses',
                levels=[0.1, 0.2, 0.3, 0.6, 0.9], ax=ax3, linewidths=3)
    circle3_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle3_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax3.add_patch(circle3_in)
    ax3.add_patch(circle3_out)
    beam3 = Circle((4, -4), radius=fwhm3, facecolor='none', ls='-',
                        edgecolor='w', linewidth=1.5, alpha=1, zorder=3)
    ax3.add_patch(beam3)
    ax3.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax3.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax3.set_title(title3, fontsize=15)
    ax3.set_xlabel(xlabel_t, fontsize=15)
    ax3.set(ylabel=None)
    ax3.set_xlim(xmin_t, xmax_t)
    ax3.set_ylim(ymin_t, ymax_t)
    ax3.tick_params(axis='x', which='major', labelsize=11)
    ax3.yaxis.set_ticks([])
    ax3.yaxis.set_ticklabels([])
    # ax3.axes.set_aspect('equal')
    ax3.plot([2*xmax_t], [2*xmax_t], color='limegreen', ls='-',
             label='non-SF stellar particles', lw=2)
    ax3.legend(facecolor='whitesmoke', framealpha=1, fontsize=13)
    axins = inset_axes(ax3, width='5%', height='100%', loc='lower left',
                        bbox_to_anchor=(1.05, 0., 1, 1),
                        bbox_transform=ax3.transAxes, borderpad=0)
    cbar = plt.colorbar(image3, cax=axins)
    cbar.set_label(label, fontsize=15)
    
    ax4.plot(xs, med1, 'k:')
    ax4.plot(xs, main1, 'k-')
    ax4.fill_between(xs, lo1, hi1, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax4.set_yscale('log')
    ax4.set_xlabel(xlabel_b, fontsize=15)
    ax4.set_ylabel(ylabel_b, fontsize=15)
    ax4.set_xlim(xmin_b, xmax_b)
    ax4.set_ylim(ymin_b, ymax_b)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax4.xaxis.set_ticks([0, 1, 2, 3, 4])
    ax4.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    # ax4.axes.set_aspect('equal')
    
    ax5.plot(xs, med2, 'k:')
    ax5.plot(xs, main2, 'k-')
    ax5.fill_between(xs, lo2, hi2, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax5.set_yscale('log')
    ax5.set_xlabel(xlabel_b, fontsize=15)
    ax5.set(ylabel=None)
    ax5.set_xlim(xmin_b, xmax_b)
    ax5.set_ylim(ymin_b, ymax_b)
    ax5.tick_params(axis='x', which='major', labelsize=11)
    ax5.tick_params(axis='y', which='minor', left=False)
    ax5.xaxis.set_ticks([0, 1, 2, 3, 4])
    ax5.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    ax5.yaxis.set_ticks([])
    ax5.yaxis.set_ticklabels([])
    # ax5.axes.set_aspect('equal')
    
    ax6.plot(xs, med3, 'k:')
    ax6.plot(xs, main3, 'k-')
    ax6.fill_between(xs, lo3, hi3, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax6.set_yscale('log')
    ax6.set_xlabel(xlabel_b, fontsize=15)
    ax6.set(ylabel=None)
    ax6.set_xlim(xmin_b, xmax_b)
    ax6.set_ylim(ymin_b, ymax_b)
    ax6.tick_params(axis='x', which='major', labelsize=11)
    ax6.tick_params(axis='y', which='minor', left=False)
    ax6.xaxis.set_ticks([0, 1, 2, 3, 4, 5])
    ax6.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'])
    ax6.yaxis.set_ticks([])
    ax6.yaxis.set_ticklabels([])
    # ax6.axes.set_aspect('equal')
    
    # gs.tight_layout(fig)
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_chains(samples, ndim, labels, save=False, outfile=None) :
    
    global currentFig
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10), sharex=True,
                                        clear=True)
    currentFig += 1
    
    ax1.plot(samples[:, :, 0], 'k', alpha=0.3)
    ax1.set_xlim(0, len(samples))
    ax1.set_ylabel(labels[0], fontsize=15)
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    
    ax2.plot(samples[:, :, 1], 'k', alpha=0.3)
    ax2.set_xlim(0, len(samples))
    ax2.set_ylabel(labels[1], fontsize=15)
    ax2.yaxis.set_label_coords(-0.1, 0.5)
    
    ax3.plot(samples[:, :, 2], 'k', alpha=0.3)
    ax3.set_xlim(0, len(samples))
    ax3.set_ylabel(labels[2], fontsize=15)
    ax3.yaxis.set_label_coords(-0.1, 0.5)
    ax3.set_xlabel('step number', fontsize=15)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_comparisons(tng_image, tng_contour_image, tng_levels,
                     skirt_image, skirt_contour_image, skirt_levels,
                     processed_image, processed_contour_image, processed_levels,
                     XX, YY, X_cent, Y_cent,
                     tng_vmin=None, tng_vmax=None,
                     skirt_vmin=None, skirt_vmax=None,
                     pro_vmin=None, pro_vmax=None,
                     xlabel=None, ylabel=None, mtitle=None,
                     xmin=None, xmax=None, ymin=None, ymax=None,
                     save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(30, 10))
    currentFig += 1
    plt.clf()
    
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0)
    
    # gs00 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax1.pcolormesh(XX, YY, tng_image, cmap=cmap,
                   norm=LogNorm(vmin=tng_vmin, vmax=tng_vmax))
    ax1.contour(X_cent, Y_cent, tng_contour_image, colors='lime',
                levels=tng_levels, linewidths=3)
    circle1_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle1_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax1.add_patch(circle1_in)
    ax1.add_patch(circle1_out)
    # ax1.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    # ax1.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax1.set_title('raw from TNG', fontsize=13)
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.set_ylabel(ylabel, fontsize=15)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    ax2.pcolormesh(XX, YY, skirt_image, cmap=cmap,
                   norm=LogNorm(vmin=skirt_vmin, vmax=skirt_vmax))
    ax2.contour(X_cent, Y_cent, skirt_contour_image, colors='lime',
                levels=skirt_levels, linewidths=3)
    circle2_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle2_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax2.add_patch(circle2_in)
    ax2.add_patch(circle2_out)
    # ax2.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    # ax2.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax2.set_title('raw from SKIRT: CASTOR UV + Roman F184', fontsize=13)
    ax2.set_xlabel(xlabel, fontsize=15)
    ax2.set(ylabel=None)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.tick_params(axis='x', which='major', labelsize=11)
    ax2.yaxis.set_ticks([])
    ax2.yaxis.set_ticklabels([])
    
    image3 = ax3.pcolormesh(XX, YY, processed_image, cmap=cmap,
                            norm=LogNorm(vmin=pro_vmin, vmax=pro_vmax))
    ax3.contour(X_cent, Y_cent, processed_contour_image, colors='lime',
                levels=processed_levels, linewidths=3)
    circle3_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle3_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax3.add_patch(circle3_in)
    ax3.add_patch(circle3_out)
    # ax3.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    # ax3.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax3.set_title('processed: CASTOR UV + Roman F184', fontsize=13)
    ax3.set_xlabel(xlabel, fontsize=15)
    ax3.set(ylabel=None)
    ax3.set_xlim(xmin, xmax)
    ax3.set_ylim(ymin, ymax)
    ax3.tick_params(axis='x', which='major', labelsize=11)
    ax3.yaxis.set_ticks([])
    ax3.yaxis.set_ticklabels([])
    # ax3.plot([2*xmax_t], [2*xmax_t], color='limegreen', ls='-',
    #           label='non-SF stellar particles', lw=2)
    # ax3.legend(facecolor='whitesmoke', framealpha=1, fontsize=13)
    # axins = inset_axes(ax3, width='5%', height='100%', loc='lower left',
    #                     bbox_to_anchor=(1.05, 0., 1, 1),
    #                     bbox_transform=ax3.transAxes, borderpad=0)
    # cbar = plt.colorbar(image3, cax=axins)
    # cbar.set_label(vlabel, fontsize=15)
    
    plt.suptitle(mtitle, fontsize=20)
    
    # gs.tight_layout(fig)
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_comprehensive_plot(title1, title2, title3, hist1, hist2, hist3,
                            contour1, contour2, contour3, level1, level2, level3,
                            xs, main1, main2, main3,
                            lo1, lo2, lo3, med1, med2, med3, hi1, hi2, hi3, XX, YY, X_cent, Y_cent,
                            times, sm, lo_sm, hi_sm, tonset, tterm, thirtySeven, seventyFive,
                            SMH, UVK_X_cent, UVK_Y_cent, UVK_contour, UVK_levels,
                            UVK_snaps_xs, UVK_snaps_ys, mtitle=None,
                            xlabel_t=None, ylabel_t=None, xlabel_b=None, ylabel_b=None,
                            xlabel_SFH=None, ylabel_SFH=None, xlabel_SMH=None, ylabel_SMH=None,
                            xlabel_UVK=None, ylabel_UVK=None, vmin=None, vmax=None,
                            xmin_t=None, xmax_t=None, ymin_t=None, ymax_t=None,
                            xmin_b=None, xmax_b=None, ymin_b=None, ymax_b=None,
                            xmin_SFH=None, xmax_SFH=None, ymin_SFH=None, ymax_SFH=None,
                            xmin_SMH=None, xmax_SMH=None, ymin_SMH=None, ymax_SMH=None,
                            xmin_UVK=None, xmax_UVK=None, ymin_UVK=None, ymax_UVK=None,
                            save=False, outfile=None, vlabel=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(25, 10))
    currentFig += 1
    plt.clf()
    
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.6, 0.4], wspace=0.25)
    
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], wspace=0)
    ax1 = fig.add_subplot(gs00[0, 0])
    ax2 = fig.add_subplot(gs00[0, 1])
    ax3 = fig.add_subplot(gs00[0, 2])
    ax4 = fig.add_subplot(gs00[1, 0])
    ax5 = fig.add_subplot(gs00[1, 1])
    ax6 = fig.add_subplot(gs00[1, 2])
    
    gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1],
                                            width_ratios=[0.4, 0.6], wspace=0.4)
    ax7 = fig.add_subplot(gs01[0, :])
    ax8 = fig.add_subplot(gs01[1, 0])
    ax9 = fig.add_subplot(gs01[1, 1])
    
    norm=LogNorm(vmin=vmin, vmax=vmax)
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax1.pcolormesh(XX, YY, hist1, cmap=cmap, norm=norm)
    ax1.contour(X_cent, Y_cent, contour1, colors='lime', levels=level1,
               linewidths=3)
    circle1_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle1_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax1.add_patch(circle1_in)
    ax1.add_patch(circle1_out)
    ax1.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax1.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax1.set_title(title1, fontsize=13)
    ax1.set_xlabel(xlabel_t, fontsize=15)
    ax1.set_ylabel(ylabel_t, fontsize=15)
    ax1.set_xlim(xmin_t, xmax_t)
    ax1.set_ylim(ymin_t, ymax_t)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    # ax1.axes.set_aspect('equal')
    
    ax2.pcolormesh(XX, YY, hist2, cmap=cmap, norm=norm)
    ax2.contour(X_cent, Y_cent, contour2, colors='lime', levels=level2,
               linewidths=3)
    circle2_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle2_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax2.add_patch(circle2_in)
    ax2.add_patch(circle2_out)
    ax2.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax2.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax2.set_title(title2, fontsize=13)
    ax2.set_xlabel(xlabel_t, fontsize=15)
    ax2.set(ylabel=None)
    ax2.set_xlim(xmin_t, xmax_t)
    ax2.set_ylim(ymin_t, ymax_t)
    ax2.tick_params(axis='x', which='major', labelsize=11)
    ax2.yaxis.set_ticks([])
    ax2.yaxis.set_ticklabels([])
    # ax2.axes.set_aspect('equal')
    
    image3 = ax3.pcolormesh(XX, YY, hist3, cmap=cmap, norm=norm)
    ax3.contour(X_cent, Y_cent, contour3, colors='lime', levels=level3,
               linewidths=3)
    circle3_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle3_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax3.add_patch(circle3_in)
    ax3.add_patch(circle3_out)
    ax3.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax3.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax3.set_title(title3, fontsize=13)
    ax3.set_xlabel(xlabel_t, fontsize=15)
    ax3.set(ylabel=None)
    ax3.set_xlim(xmin_t, xmax_t)
    ax3.set_ylim(ymin_t, ymax_t)
    ax3.tick_params(axis='x', which='major', labelsize=11)
    ax3.yaxis.set_ticks([])
    ax3.yaxis.set_ticklabels([])
    # ax3.axes.set_aspect('equal')
    ax3.plot([2*xmax_t], [2*xmax_t], color='limegreen', ls='-',
             label='non-SF stellar particles', lw=2)
    ax3.legend(facecolor='whitesmoke', framealpha=1, fontsize=13)
    axins = inset_axes(ax3, width='5%', height='100%', loc='lower left',
                        bbox_to_anchor=(1.05, 0., 1, 1),
                        bbox_transform=ax3.transAxes, borderpad=0)
    cbar = plt.colorbar(image3, cax=axins)
    cbar.set_label(vlabel, fontsize=15)
    
    ax4.plot(xs, med1, 'k:')
    ax4.plot(xs, main1, 'ro')
    ax4.fill_between(xs, lo1, hi1, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax4.set_yscale('log')
    ax4.set_xlabel(xlabel_b, fontsize=15)
    ax4.set_ylabel(ylabel_b, fontsize=15)
    ax4.set_xlim(xmin_b, xmax_b)
    ax4.set_ylim(ymin_b, ymax_b)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax4.xaxis.set_ticks([0, 1, 2, 3, 4])
    ax4.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    # ax4.axes.set_aspect('equal')
    
    ax5.plot(xs, med2, 'k:')
    ax5.plot(xs, main2, 'ro')
    ax5.fill_between(xs, lo2, hi2, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax5.set_yscale('log')
    ax5.set_xlabel(xlabel_b, fontsize=15)
    ax5.set(ylabel=None)
    ax5.set_xlim(xmin_b, xmax_b)
    ax5.set_ylim(ymin_b, ymax_b)
    ax5.tick_params(axis='x', which='major', labelsize=11)
    ax5.tick_params(axis='y', which='minor', left=False)
    ax5.xaxis.set_ticks([0, 1, 2, 3, 4])
    ax5.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    ax5.yaxis.set_ticks([])
    ax5.yaxis.set_ticklabels([])
    # ax5.axes.set_aspect('equal')
    
    ax6.plot(xs, med3, 'k:')
    ax6.plot(xs, main3, 'ro')
    ax6.fill_between(xs, lo3, hi3, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax6.set_yscale('log')
    ax6.set_xlabel(xlabel_b, fontsize=15)
    ax6.set(ylabel=None)
    ax6.set_xlim(xmin_b, xmax_b)
    ax6.set_ylim(ymin_b, ymax_b)
    ax6.tick_params(axis='x', which='major', labelsize=11)
    ax6.tick_params(axis='y', which='minor', left=False)
    ax6.xaxis.set_ticks([0, 1, 2, 3, 4, 5])
    ax6.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'])
    ax6.yaxis.set_ticks([])
    ax6.yaxis.set_ticklabels([])
    # ax6.axes.set_aspect('equal')
    
    ax7.plot(times, sm, color='k', label='SFH', marker='', linestyle='-',
             alpha=1)
    ax7.plot(times, lo_sm, color='grey', label='lo, hi', marker='', linestyle='-.',
             alpha=0.8)
    ax7.plot(times, hi_sm, color='grey', label='', marker='', linestyle='-.',
             alpha=0.8)
    if (ymin_SFH == None) :
        _, _, ymin_SFH, _ = ax7.axis()
    if (ymax_SFH == None) :
        _, _, _, ymax_SFH = ax7.axis()
    cmap_SFH = mcol.LinearSegmentedColormap.from_list('BlRd',['b','r'])
    ax7.imshow([[0.,1.], [0.,1.]], extent=(tonset, tterm, ymin_SFH, ymax_SFH),
                cmap=cmap_SFH, interpolation='bicubic', alpha=0.15, aspect='auto')
    ax7.axvline(thirtySeven, color='k', ls=':')
    ax7.axvline(seventyFive, color='k', ls='--')
    ax7.axvline(tonset, color='b', ls=':', alpha=0.15) # label=r'$t_{\rm onset}$'
    ax7.axvline(tterm, color='r', ls=':', alpha=0.15) # label=r'$t_{\rm termination}$'
    ax7.set_title(r'$\Delta t_{\rm quench} = $' + '{:.1f} Gyr'.format(tterm-tonset),
                  fontsize=13)
    ax7.set_xlabel(xlabel_SFH, fontsize=15)
    ax7.set_ylabel(ylabel_SFH, fontsize=15)
    ax7.set_xlim(xmin_SFH, xmax_SFH)
    ax7.set_ylim(ymin_SFH, ymax_SFH)
    ax7.legend(facecolor='whitesmoke', framealpha=1, fontsize=13, loc=0)
    
    ax8.plot(times, SMH, color='k', label='SMH', marker='', linestyle='-',
             alpha=1)
    if (ymin_SMH == None) :
        _, _, ymin_SMH, _ = ax8.axis()
    if (ymax_SMH == None) :
        _, _, _, ymax_SFH = ax8.axis()
    cmap_SFH = mcol.LinearSegmentedColormap.from_list('BlRd',['b','r'])
    ax8.imshow([[0.,1.], [0.,1.]], extent=(tonset, tterm, ymin_SFH, ymax_SFH),
                cmap=cmap_SFH, interpolation='bicubic', alpha=0.15, aspect='auto')
    ax8.axvline(thirtySeven, color='k', ls=':')
    ax8.axvline(seventyFive, color='k', ls='--')
    ax8.axvline(tonset, color='b', ls=':', alpha=0.15) # label=r'$t_{\rm onset}$'
    ax8.axvline(tterm, color='r', ls=':', alpha=0.15) # label=r'$t_{\rm termination}$'
    ax8.set_xlabel(xlabel_SMH, fontsize=15)
    ax8.set_ylabel(ylabel_SMH, fontsize=15)
    ax8.set_xlim(xmin_SMH, xmax_SMH)
    ax8.set_ylim(ymin_SMH, ymax_SMH)
    ax8.legend(facecolor='whitesmoke', framealpha=1, fontsize=13, loc=0)
    
    ax9.contour(UVK_X_cent, UVK_Y_cent, UVK_contour, colors='grey',
                levels=UVK_levels, linewidths=1) # levels=[0.1, 0.3, 0.5, 0.7, 0.9]
    ax9.scatter(UVK_snaps_xs, UVK_snaps_ys, c=['blue', 'purple', 'red', 'k'],
                marker='o', edgecolors='grey', s=40, zorder=3)
    ax9.set_xlabel(xlabel_UVK, fontsize=15)
    ax9.set_ylabel(ylabel_UVK, fontsize=15)
    ax9.set_xlim(xmin_UVK, xmax_UVK)
    ax9.set_ylim(ymin_UVK, ymax_UVK)
    
    plt.suptitle(mtitle, fontsize=20)
    
    # gs.tight_layout(fig)
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_lines(xx, yy, xs, ys, label=None,
               xlabel=None, ylabel=None, xmin=None, xmax=None,
               ymin=None, ymax=None, save=False, outfile=None,
               figsizewidth=9.5, figsizeheight=7, loc=0) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.plot(xx, yy, '-', color='grey')
    ax.scatter(xx, yy, s=100, linestyles='-', color=['b', 'purple', 'r'],
               edgecolors='k', zorder=5, label=label)
    
    for i, (x_indv, y_indv) in enumerate(zip(xs, ys)) :
        if i == 0 :
            slabel = 'control sample'
        else :
            slabel = ''
        # ax.plot(x_indv, y_indv, '-', color='grey', alpha=0.2)
        ax.scatter(x_indv[1], y_indv[1], linestyles='-', label=slabel,
                   # color=['b', 'purple', 'r'],
                   color='purple',
                   alpha=0.3, zorder=5)
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter(xs, ys, color, label, marker, cbar_label='', size=30,
                 xlabel=None, ylabel=None, title=None, cmap=cm.rainbow,
                 xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                 figsizewidth=9.5, figsizeheight=7, scale='linear',
                 vmin=None, vmax=None, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    # cmap = copy.copy(cmap)
    # norm = Normalize(vmin=vmin, vmax=vmax)
    
    # temp = np.linspace(min(np.nanmin(xs), np.nanmin(ys)),
    #                    max(np.nanmax(xs), np.nanmax(ys)), 1000)
    temp = np.linspace(xmin, xmax, 1000)
    ax.plot(temp, temp, 'k-')
    # ax.plot(temp, temp+0.5, 'k--')
    # ax.plot(temp, temp+1, 'k:')
    
    ax.scatter(xs, ys, c=color, marker='o', alpha=0.2)
    
    # frame = ax.scatter(xs, ys, c=color, marker=marker, label=label, cmap=cmap,
                        # edgecolors='grey',
                       # norm=norm, s=size, alpha=0.3)
    # cbar = plt.colorbar(frame)
    # cbar.set_label(cbar_label, fontsize=15)
    # ax.plot(-5, -5, 'o', c=cmap(327), alpha=0.3, label='cluster')
    # ax.plot(-5, -5, 'o', c=cmap(170), alpha=0.3, label='high mass group')
    # ax.plot(-5, -5, 'o', c=cmap(100), alpha=0.3, label='low mass group')
    # ax.plot(-5, -5, 'o', c=cmap(0), alpha=0.3, label='field')
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_dumb(xs, ys, color, label, marker, cbar_label='', size=30,
                      xlabel=None, ylabel=None, title=None, cmap=cm.rainbow,
                      xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                      figsizewidth=9.5, figsizeheight=7, scale='linear',
                      vmin=None, vmax=None, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    # norm = Normalize(vmin=vmin, vmax=vmax)
    
    if (xmin is None) and (xmax is None) :
        xmin, xmax = np.min(xs), np.max(xs)
    xx = np.linspace(xmin, xmax, 1000)
    ax.plot(xx, xx, 'k-', label='equality')
    frame = ax.scatter(xs, ys, c=color, marker=marker, label=label, cmap=cmap,
                        # edgecolors='grey', norm=norm,
                       vmin=vmin, vmax=vmax, s=size, alpha=0.5, zorder=3)
    cbar = plt.colorbar(frame)
    cbar.set_label(cbar_label, fontsize=15)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_err(xs, ys, lo, hi, xlabel=None, ylabel=None,
                     xmin=None, xmax=None, ymin=None, ymax=None,
                     figsizewidth=9.5, figsizeheight=7, save=False,
                     outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.errorbar(xs, ys, xerr=[lo, hi], fmt='ko', ecolor='k', elinewidth=0.5,
                capsize=2)
    
    ax.axhline(50, c='grey', ls='--')
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_err_both(xs, ys, xlo, xhi, ylo, yhi, xlabel=None, ylabel=None,
                          xmin=None, xmax=None, ymin=None, ymax=None,
                          figsizewidth=9.5, figsizeheight=7, save=False,
                          outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.errorbar(xs, ys, xerr=[xlo, xhi], yerr=[ylo, yhi], fmt='ko', ecolor='k',
                elinewidth=0.5, capsize=2)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_multi(xs, ys, colors, labels, markers, alphas,
                       xlabel=None, ylabel=None, title=None, cmap=cm.rainbow,
                       xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                       figsizewidth=9.5, figsizeheight=7, scale='linear',
                       vmin=None, vmax=None, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    for i in range(len(xs)) :
        ax.scatter(xs[i], ys[i], c=colors[i], marker=markers[i], label=labels[i],
                   cmap=cmap, norm=norm, alpha=alphas[i])
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_multi_with_bands(SF_xs, SF_ys, q_xs, q_ys, other_xs, other_ys,
                                  centers, lo, hi, xlabel=None, ylabel=None,
                                  xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                                  figsizewidth=9.5, figsizeheight=7,
                                  save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.fill_between(centers, lo, hi, color='grey', alpha=0.2,
                    edgecolor='darkgrey')
    ax.scatter(SF_xs, SF_ys, c='b', marker='o', label='SFMS', edgecolors='grey',
               alpha=0.1)
    ax.scatter(q_xs, q_ys, c='r', marker='o', label='', edgecolors='grey',
               alpha=0.1)
    ax.scatter(other_xs, other_ys, c='k', marker='o', label='', edgecolors='grey',
               alpha=0.2)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_particles(dx, dy, dz, sf_dx, sf_dy, sf_dz, xlabel=None,
                           ylabel=None, zlabel=None, figsizewidth=18,
                           figsizeheight=6, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    ax1.scatter(dx, dy, color='r', alpha=0.05)
    ax1.scatter(sf_dx, sf_dy, color='b', alpha=0.1)
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.set_ylabel(ylabel, fontsize=15)
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-30, 30)
    
    ax2.scatter(dx, dz, color='r', alpha=0.05)
    ax2.scatter(sf_dx, sf_dz, color='b', alpha=0.1)
    ax2.set_xlabel(xlabel, fontsize=15)
    ax2.set_ylabel(zlabel, fontsize=15)
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-30, 30)
    
    ax3.scatter(dy, dz, color='r', alpha=0.05)
    ax3.scatter(sf_dy, sf_dz, color='b', alpha=0.1)
    ax3.set_xlabel(ylabel, fontsize=15)
    ax3.set_ylabel(zlabel, fontsize=15)
    ax3.set_xlim(-30, 30)
    ax3.set_ylim(-30, 30)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_with_bands(xs, ys, centers, lo, hi,
                            xlabel=None, ylabel=None,
                            xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                            figsizewidth=9.5, figsizeheight=7,
                            save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.fill_between(centers, lo, hi, color='grey', alpha=0.2,
                    edgecolor='darkgrey')
    ax.scatter(xs, ys, c='b', marker='o', label='', edgecolors='grey', alpha=0.01)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_with_hists(xs, ys, colors, labels, markers, alphas,
                            xlabel=None, ylabel=None, title=None, loc=0,
                            xmin=None, xmax=None, ymin=None, ymax=None,
                            figsizewidth=9.5, figsizeheight=7, save=False,
                            outfile=None, xbins=None, ybins=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    
    for xx, yy, color, label, marker, alpha in zip(xs, ys, colors, labels,
                                                   markers, alphas) :
        
        ax.scatter(xx, yy, c=color, marker=marker, label=label, alpha=alpha)
        if alpha > 0.5 : # alpha is capped at 1
            alpha = 0.5
        if marker != '' :
            ax_histx.hist(xx, bins=xbins, color=color, histtype='step',
                          alpha=2*alpha, weights=np.ones(len(xx))/len(xx))
            ax_histy.hist(yy, bins=ybins, color=color, histtype='step',
                          alpha=2*alpha, weights=np.ones(len(yy))/len(yy),
                          orientation='horizontal')
    
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
    ax_histx.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax_histx.set_ylim(0, 0.8)
    ax_histy.set_xlim(0, 0.3)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_sed(waves, sed, waves_phot, phot_model, phot, phot_e,
             xlabel=None, ylabel=None, title=None,
             xmin=0.13, xmax=3, ymin=0.1, ymax=10,
             figsizewidth=12, figsizeheight=9, loc=4,
             outfile=None, save=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    # ax.plot(waves, direct, color='r', marker='', ls=':', lw=3, alpha=0.3,
    #         label='FAST++ output')
    ax.plot(waves, sed, color='k', marker='', ls='-', alpha=0.3,
            label='from SFH parameters')
    ax.plot(waves_phot, phot_model, color='r', marker='s', ls='',
            label='model flux', mfc='none', mew=2, ms=12)
    # ax.plot(waves_phot, phot, color='b', marker='o', ls='',
    #         label='observed flux', mfc='none', mew=2, ms=12)
    ax.errorbar(waves_phot, phot, yerr=phot_e, xerr=None, color='b', marker='o',
                ls='', label='observed flux', mfc='none', mew=2, ms=12,
                ecolor='b', elinewidth=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('wavelength (micron)', fontsize=15)
    ax.set_ylabel(r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
                  fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_seds(waves, seds, waves_phot, phot_model, phot, phot_e,
              xlabel=None, ylabel=None, title=None,
              xmin=0.13, xmax=3, ymin=0.1, ymax=10,
              figsizewidth=12, figsizeheight=9, loc=4,
              outfile=None, save=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    from matplotlib.collections import LineCollection
    
    colors = cm.magma(np.arange(len(seds))/len(seds))
    # colors = cm.magma(np.arange(3)/3).tolist() + [[0, 0, 1, 1]]
    # labels = ['constructed', 'FAST++ output'] + ['']*(len(seds) - 2)
    labels = ['']*len(seds)
    # labels = ['bestfit (extended parameter space)',
    #           'previous bestfit (limited parameter space)',
    #           r'best $Z = Z_{\odot}$ fit'] + ['']*(len(seds) - 3)
    # ls = ['--', '-', '-', '-']
    # al = [0.3, 0.3, 0.3, 0.5]
    # ax.plot(waves, direct, color='r', marker='', ls=':', lw=3, alpha=0.3,
    #         label='FAST++ output')
    for i in range(len(seds)) :
        # [ax.plot(waves[10*j:10*(j+1)+1], seds[i][10*j:10*(j+1)+1],
        #  alpha=np.max([0.7-0.015*j, 0.25]), color=colors[i], marker='', ls='-')
        #  for j in range(int(len(waves)/10))]
        ax.plot(waves, seds[i], color=colors[i], marker='', ls='-', alpha=0.3,
                label=labels[i])
    ax.plot(waves_phot, phot_model, color='r', marker='s', ls='',
            label='model flux', mfc='none', mew=2, ms=12)
    # ax.plot(waves_phot, phot, color='b', marker='o', ls='',
    #         label='observed flux', mfc='none', mew=2, ms=12)
    ax.errorbar(waves_phot, phot, yerr=phot_e, xerr=None, color='b', marker='o',
                ls='', label='observed flux', mfc='none', mew=2, ms=12,
                ecolor='b', elinewidth=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('wavelength (micron)', fontsize=15)
    ax.set_ylabel(r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
                  fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_sfhs_comparison(xs, ys, time, data, centers,
                         xlabel=None, ylabel=None,
                         xmin=None, xmax=None, ymin=None, ymax=None,
                         figsizewidth=9.5, figsizeheight=7, loc=0) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.plot(xs, ys, 'k-')
    ax.axvline(time, color='k', ls=':', label='observation', alpha=0.5)
    
    colors = cm.hot(np.arange(len(data))/len(data))
    for i in range(len(data)) :
        if i in [0, len(data) - 1] :
            label = 'bin {}'.format(i)
        else :
            label = ''
        ax.plot(centers, data[i], color=colors[i], ls='--', label=label)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    plt.show()
    
    return

def plot_simple_dumb(xs, ys, label='', save=False,
                     xlabel=None, ylabel=None, title=None, outfile=None,
                     xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                     figsizewidth=9.5, figsizeheight=7, scale='linear') :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    # xx = np.linspace(xmin, xmax, 1000)
    # ax.plot(xx, xx, 'r-', label='equality')
    # ax.plot(xx, xx+0.18, 'b-', label='y = x + 0.18')
    ax.plot(xs, ys, 'k-', label=label) #alpha=0.2)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if label != '' :
        ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
                      xlabel=None, ylabel=None, title=None,
                      xmin=None, xmax=None, ymin=None, ymax=None,
                      figsizewidth=9.5, figsizeheight=7, scale='linear', loc=0,
                      outfile=None, save=False, ms=6) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for i in range(len(xs)) :
        ax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                color=colors[i], label=labels[i], alpha=alphas[i], ms=ms,
                # mfc='none',
                mec=colors[i], mew=2)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # if labels[0] != '' :
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_stairs(values, edges) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(9.5, 7))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.stairs(values, edges, color='k', baseline=None)
    
    plt.tight_layout()
    plt.show()
    
    return

def plot_multi_vertical_error(xs, ys, labels, colors, markers, styles,
                              reg, xlabel=None, ylabel1=None, ylabel2=None,
                              ylabel3=None, xmin=None, xmax=None, ymin1=None,
                              ymax1=None, ymin2=None, ymax2=None, ymin3=None,
                              ymax3=None, figsizewidth=9.5, figsizeheight=9.5,
                              loc=0, save=False, outfile=None, title=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    
    gs = fig.add_gridspec(3, 1, hspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    
    # alphas = [0.5, 0.65, 1]*3
    # lws = [1.25, 1.5, 2.5]*3
    alphas = [0.5, 0.65]*3
    lws = [1.25, 1.5]*3
    
    for i in range(reg) :
        ax1.plot(xs[i], ys[i], #yerr=[lo[i], hi[i]],
                     marker=markers[i],
                     linestyle=styles[i], color=colors[i], label=labels[i],
                     alpha=alphas[i], lw=lws[i])
                     # ecolor='lightgray', elinewidth=1.5)
    
    for i in range(reg, 2*reg) :
        ax2.plot(xs[i], ys[i], #yerr=[lo[i], hi[i]],
                     marker=markers[i],
                     linestyle=styles[i], color=colors[i], label=labels[i],
                     alpha=alphas[i], lw=lws[i])
                     # ecolor='lightgray', elinewidth=1.5)
    
    for i in range(2*reg, len(xs)) :
        ax3.plot(xs[i], ys[i], #yerr=[lo[i], hi[i]],
                     marker=markers[i],
                     linestyle=styles[i], color=colors[i], label=labels[i],
                     alpha=alphas[i], lw=lws[i])
                     # ecolor='lightgray', elinewidth=1.5)
    
    ax1.tick_params(bottom=False, labelbottom=False)
    ax2.tick_params(bottom=False, labelbottom=False)
    
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    
    ax1.set_title(title, fontsize=10)
    ax3.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel(ylabel1, fontsize=10)
    ax2.set_ylabel(ylabel2, fontsize=10)
    ax3.set_ylabel(ylabel3, fontsize=10)
    
    ax2.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin1, ymax1)
    ax2.set_ylim(ymin2, ymax2)
    ax3.set_ylim(ymin3, ymax3)
    
    if ax2.get_ylim()[0] < 1e-10 :
        ax2.set_ylim(1e-10, ymax2)
    
    if ax3.get_ylim()[0] < 1e-15 :
        ax3.set_ylim(1e-15, ymax3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10,
               facecolor='whitesmoke', framealpha=1, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_multi_error(xs, ys, lo, hi, labels, colors, markers, styles, reg,
                     xlabel=None, ylabel=None, seclabel=None, xmin=None,
                     xmax=None, ymin=None, ymax=None, secmin=None, secmax=None,
                     figsizewidth=9.5, figsizeheight=7, loc=0, save=False,
                     outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    secax = ax.twinx()
    secax.format_coord = make_format(secax, ax)
    
    for i in range(reg) :
        ax.errorbar(xs[i], ys[i], yerr=[lo[i], hi[i]], marker=markers[i],
                    linestyle=styles[i], color=colors[i], label=labels[i],
                    ecolor=colors[i], elinewidth=1.5)
    
    for i in range(reg, len(xs)) :
        secax.errorbar(xs[i], ys[i], yerr=[lo[i], hi[i]], marker=markers[i],
                       linestyle=styles[i], color=colors[i], label=labels[i],
                       ecolor=colors[i], elinewidth=1.5)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    secax.set_ylabel(seclabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    secax.set_ylim(secmin, secmax)
    
    lines, labels = ax.get_legend_handles_labels()
    seclines, seclabels = secax.get_legend_handles_labels()
    
    ax.legend(lines + seclines, labels + seclabels, fontsize=15,
              facecolor='whitesmoke', framealpha=1, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_multi_secax(xs, ys, labels, colors, markers, styles,
                            alphas, reg, xlabel=None, ylabel=None,
                            seclabel=None, title=None, xmin=None,
                            xmax=None, ymin=None, ymax=None,
                            secmin=None, secmax=None, loc=0,
                            figsizewidth=9.5, figsizeheight=7,
                            xscale='linear', yscale='log',
                            secscale='linear', outfile=None,
                            save=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    secax = ax.twinx()
    secax.format_coord = make_format(secax, ax)
    
    for i in range(reg) :
        ax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                color=colors[i], label=labels[i], alpha=alphas[i])
    
    for i in range(reg, len(xs)) :
        secax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                   color=colors[i], label=labels[i], alpha=alphas[i])
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    secax.set_yscale(secscale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    secax.set_ylabel(seclabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    secax.set_ylim(secmin, secmax)
    
    lines, labels = ax.get_legend_handles_labels()
    seclines, seclabels = secax.get_legend_handles_labels()
    
    ax.legend(lines + seclines, labels + seclabels, fontsize=15,
              facecolor='whitesmoke', framealpha=1, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def make_format(current, other) :
    # adapted from https://stackoverflow.com/questions/21583965
    
    # current and other are axes
    def format_coord(x, y) :
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Left: {:<40}    Right: {:<}'
                .format(*['({:.3g}, {:.3g})'.format(x, y) for x,y in coords]))
    return format_coord

def plot_simple_with_band(xs, ys, lo, med, hi, xlabel=None, ylabel=None,
                          xmin=None, xmax=None, ymin=None, ymax=None,
                          figsizewidth=7, figsizeheight=7, scale='linear', loc=0,
                          outfile=None, save=False, legend=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.plot(xs, med, 'k:')
    ax.plot(xs, ys, 'k-')
    ax.fill_between(xs, lo, hi, color='grey', edgecolor='darkgrey', alpha=0.2)
    
    ax.set_xscale(scale)
    ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if legend :
        ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=18, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def test_contour(XX, YY, hist, X_cent, Y_cent, contours, levels,
                 xlabel=None, ylabel=None, save=False, outfile=None,
                 xmin=None, xmax=None, ymin=None, ymax=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(8, 8))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    norm = LogNorm(vmin=1e-11, vmax=2e-09)
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax.pcolormesh(XX, YY, hist, cmap=cmap, norm=norm, alpha=0.9)
    ax.contour(X_cent, Y_cent, contours, colors='grey', levels=levels,
               linewidths=1)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return
