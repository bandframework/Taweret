#!/usr/bin/env python3
import numpy as np
import matplotlib
import dill
#matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
import tkinter.font
from configurations import *
from emulator import Trained_Emulators, _Covariance
from bayes_exp import Y_exp_data
from bayes_plot import obs_tex_labels_2

short_names = {
                'norm' : 'N',
                'trento_p' : 'p',
                'nucleon_width' : 'w',
                'sigma_k' : 'sigma_k',
                'dmin3' : 'd_{min}^3',
                'tau_R' : 'tau_R',
                'alpha' : 'alpha',
                'eta_over_s_T_kink_in_GeV' : 'eta_Tk',
                'eta_over_s_low_T_slope_in_GeV' : 'eta_low',
                'eta_over_s_high_T_slope_in_GeV' : 'eta_high',
                'eta_over_s_at_kink' : 'eta_k',
                'zeta_over_s_max' : 'zeta_max',
                'zeta_over_s_T_peak_in_GeV' : 'zeta_Tc',
                'zeta_over_s_width_in_GeV' : 'zeta_w',
                'zeta_over_s_lambda_asymm' : 'zeta_asym',
                'shear_relax_time_factor' : 'b_pi',
                'Tswitch' : 'T_sw',
}

MAP_params = {
            'Pb-Pb-2760': [14.128, 0.089, 1.054, 1.064, 4.227, 1.507, 0.113, 0.223, -1.585, 0.32, 0.056, 0.11, 0.16, 0.093, -0.084, 4.666, 0.136],
            'Au-Au-200' : [5.821, 0.089, 1.054, 1.064, 4.227, 1.507, 0.113, 0.223, -1.585, 0.32, 0.056, 0.11, 0.16, 0.093, -0.084, 4.666, 0.136]
}

system_observables = {
                    'Pb-Pb-2760' : ['dET_deta', 'dN_dy_pion', 'dN_dy_proton', 'mean_pT_pion', 'mean_pT_proton', 'pT_fluct', 'v22', 'v32', 'v42'],
                    'Au-Au-200' : ['dN_dy_pion', 'dN_dy_kaon', 'mean_pT_pion', 'mean_pT_kaon', 'v22', 'v32']
                    }

class Application(tk.Frame):
    def __init__(self, master=None):

        self.system = 'Pb-Pb-2760'
        self.Yexp = Y_exp_data

        #load the design
        design_file = SystemsInfo[self.system]["main_design_file"]
        range_file = SystemsInfo[self.system]["main_range_file"]
        design = pd.read_csv(design_file)
        design = design.drop("idx", axis=1)
        labels = design.keys()
        design_range = pd.read_csv(range_file)
        self.design_max = design_range['max'].values
        self.design_min = design_range['min'].values

        self.Names = labels
        self.idf = 0

        #load the emulator
        self.emu = dill.load(open('emulator/emulator-' + 'Pb-Pb-2760' + '-idf-' + str(self.idf) + '.dill', "rb"))

        super().__init__(master)
        self.createWidgets()

    def toggle(self):
        self.idf = (self.idf + 1)%4
        self.change_idf.config(text='df : ' + idf_label[self.idf])
        self.emu = dill.load(open('emulator/emulator-' + 'Pb-Pb-2760' + '-idf-' + str(self.idf) + '.dill', "rb"))

    """
    def toggle_sys(self):
        if self.system == 'Pb-Pb-2760':
            self.system = 'Au-Au-200'
        else:
            self.system = 'Pb-Pb-2760'

        self.change_sys.config(text='system : ' + self.system)
        self.emu = dill.load(open('emulator/emulator-' + self.system + '-idf-' + str(self.idf) + '.dill', "rb"))

        self.observables = system_observables[self.system]
        self.nobs = len(self.observables)
    """

    def createWidgets(self):
        self.observables = system_observables[self.system]
        self.nobs = len(self.observables)
        nrows = 3
        fig, self.axes = plt.subplots(nrows=nrows, ncols= self.nobs // nrows, figsize=(6,6))

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=1,column=0, rowspan=15, columnspan=2)
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.9)
        self.canvas.draw()

        # plot
        self.plotbutton=tk.Button(master=root, text="plot (click to begin)", command=lambda: self.plot())
        self.plotbutton.grid(row=0,column=0, rowspan=1)

        # switching delta f
        self.change_idf = tk.Button(text="df : Grad(click to switch)", command=lambda: [self.toggle(), self.plot()])
        self.change_idf.grid(row=0, column=1)

        # switching system
        #self.change_sys = tk.Button(text="system : Pb-Pb-2760(click to switch)", command=lambda: [self.toggle_sys(), self.plot()])
        #self.change_sys.grid(row=0, column=3)

        label_font = tk.font.Font(family='Arial', size=10)

        for i, name in enumerate(self.Names):

            name_short = short_names[name]
            ncols = 2
            row = i // ncols
            col = i - (row * ncols)
            col += 3
            row += 1

            l = self.design_min[i]
            h = self.design_max[i]
            setattr(self, name, l)
            # add the slide scale for this variable
            setattr(self, 'tune'+name, tk.Scale(master=root, from_=l, to=h, resolution=(h-l)/50.,
            length=200, orient="horizontal", borderwidth=2.0, width=15) )

            # labelling the slide scale
            setattr(self, 'label'+name, tk.Label(master=root, text=name_short, font=label_font))
            getattr(self, 'label'+name).grid(row=row, column=col, sticky=SW)
            getattr(self, 'tune'+name).set(MAP_params[self.system][i])
            getattr(self, 'tune'+name).grid(row=row, column=col)
            getattr(self, 'tune'+name).bind("<B1-Motion>", lambda event: self.plot() )

    def formatting_plot(f):
        def ff(self):
            f(self)
            plt.tight_layout(True)
            plt.subplots_adjust(top=0.9)
            self.canvas.draw()
        return ff

    @formatting_plot
    def plot(self):

        params = [getattr(self, 'tune'+name).get() for name in self.Names]
        Yemu_mean, Yemu_cov = self.emu.predict( np.array( [params] ), return_cov=True )

        for obs, ax in zip(self.observables, self.axes.flatten()):
            ax.cla()
            ax.set_title(obs_tex_labels_2[obs])

            xbins = np.array(obs_cent_list[self.system][obs])
            #centrality bins
            x = (xbins[:,0]+xbins[:,1])/2.
            #emulator prediction
            y_emu = Yemu_mean[obs][0]
            dy_emu = (np.diagonal(np.abs(Yemu_cov[obs, obs]))**.5)[:,0]
            #ax.plot(x, y_emu, label='emu')
            ax.fill_between(x, y_emu-dy_emu, y_emu+dy_emu)
            #experiment
            exp_mean = self.Yexp[self.system][obs]['mean'][idf]
            exp_err = self.Yexp[self.system][obs]['err'][idf]
            ax.errorbar( x, exp_mean, exp_err, color='black', marker='v')

            if obs == 'dNch_deta':
                ax.set_ylim(0, 2e3)
            if obs == 'dN_dy_pion':
                ax.set_ylim(0, 2e3)
            if obs == 'dN_dy_proton':
                ax.set_ylim(0, 1e2)
            if obs == 'dET_deta':
                ax.set_ylim(0, 2.5e3)
            if obs == 'mean_pT_pion':
                ax.set_ylim(0.3, 0.7)
            if obs == 'mean_pT_proton':
                ax.set_ylim(0.75, 1.75)
            if obs == 'pT_fluct':
                ax.set_ylim(0, 0.04)
            if obs == 'v22':
                ax.set_ylim(0, 0.13)
            if obs == 'v32':
                ax.set_ylim(0.01, 0.04)
            if obs == 'v42':
                ax.set_ylim(0.005, 0.018)

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.master.title('Hand tuning your parameters')
    app.mainloop()
