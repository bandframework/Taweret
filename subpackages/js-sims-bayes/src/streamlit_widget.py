import streamlit as st
import numpy as np
import time
import os
import subprocess
import matplotlib
import altair as alt
import dill
import matplotlib.pyplot as plt
from configurations import *
from emulator import Trained_Emulators, _Covariance
from bayes_exp import Y_exp_data
from bayes_plot import obs_tex_labels_2

# https://gist.github.com/beniwohli/765262
greek_alphabet_inv = { u'\u0391': 'Alpha', u'\u0392': 'Beta', u'\u0393': 'Gamma', u'\u0394': 'Delta', u'\u0395': 'Epsilon', u'\u0396': 'Zeta', u'\u0397': 'Eta', u'\u0398': 'Theta', u'\u0399': 'Iota', u'\u039A': 'Kappa', u'\u039B': 'Lamda', u'\u039C': 'Mu', u'\u039D': 'Nu', u'\u039E': 'Xi', u'\u039F': 'Omicron', u'\u03A0': 'Pi', u'\u03A1': 'Rho', u'\u03A3': 'Sigma', u'\u03A4': 'Tau', u'\u03A5': 'Upsilon', u'\u03A6': 'Phi', u'\u03A7': 'Chi', u'\u03A8': 'Psi', u'\u03A9': 'Omega', u'\u03B1': 'alpha', u'\u03B2': 'beta', u'\u03B3': 'gamma', u'\u03B4': 'delta', u'\u03B5': 'epsilon', u'\u03B6': 'zeta', u'\u03B7': 'eta', u'\u03B8': 'theta', u'\u03B9': 'iota', u'\u03BA': 'kappa', u'\u03BB': 'lamda', u'\u03BC': 'mu', u'\u03BD': 'nu', u'\u03BE': 'xi', u'\u03BF': 'omicron', u'\u03C0': 'pi', u'\u03C1': 'rho', u'\u03C3': 'sigma', u'\u03C4': 'tau', u'\u03C5': 'upsilon', u'\u03C6': 'phi', u'\u03C7': 'chi', u'\u03C8': 'psi', u'\u03C9': 'omega', }
greek_alphabet = {v: k for k, v in greek_alphabet_inv.items()}

zeta_over_s_str=greek_alphabet['zeta']+'/s(T)'
eta_over_s_str=greek_alphabet['eta']+'/s(T)'

v2_str='v'u'\u2082''{2}'
v3_str='v'u'\u2083''{2}'
v4_str='v'u'\u2084''{2}'


short_names = {
                'norm' : r'Energy Normalization', #0
                'trento_p' : r'TRENTo Reduced Thickness', #1
                'sigma_k' : r'Multiplicity Fluctuation', #2
                'nucleon_width' : r'Nucleon width [fm]', #3
                'dmin3' : r'Min. Distance btw. nucleons cubed [fm^3]', #4
                'tau_R' : r'Free-streaming time scale [fm/c]', #5
                'alpha' : r'Free-streaming energy dep.', #6
                'eta_over_s_T_kink_in_GeV' : r'Temperature of shear kink [GeV]', #7
                'eta_over_s_low_T_slope_in_GeV' : r'Low-temp. shear slope [GeV^-1]', #8
                'eta_over_s_high_T_slope_in_GeV' : r'High-temp shear slope [GeV^-1]', #9
                'eta_over_s_at_kink' : r'Shear viscosity at kink', #10
                'zeta_over_s_max' : r'Bulk viscosity max.', #11
                'zeta_over_s_T_peak_in_GeV' : r'Temperature of max. bulk viscosity [GeV]', #12
                'zeta_over_s_width_in_GeV' : r'Width of bulk viscosity [GeV]', #13
                'zeta_over_s_lambda_asymm' : r'Skewness of bulk viscosity', #14
                'shear_relax_time_factor' : r'Shear relaxation time normalization', #15
                'Tswitch' : 'Particlization temperature [GeV]', #16
}


system_observables = {
                    'Pb-Pb-2760' : ['dET_deta', 'dN_dy_pion', 'dN_dy_proton', 'mean_pT_pion', 'mean_pT_proton', 'pT_fluct', 'v22', 'v32', 'v42'],
                    'Au-Au-200' : ['dN_dy_pion', 'dN_dy_kaon', 'mean_pT_pion', 'mean_pT_kaon', 'v22', 'v32']
                    }

obs_lims = {'dET_deta' : 2500. , 'dN_dy_pion' : 2000., 'dN_dy_proton' : 100., 'mean_pT_pion' : 1., 'mean_pT_proton' : 2., 'pT_fluct' : .05, 'v22' : .2, 'v32' : .05, 'v42' :.03 }

obs_word_labels = {
                    'dNch_deta' : r'Charged multiplicity',
                    'dN_dy_pion' : r'Pion dN/dy',
                    'dN_dy_kaon' : r'Kaon dN/dy',
                    'dN_dy_proton' : r'Proton dN/dy',
                    'dN_dy_Lambda' : r'Lambda dN/dy',
                    'dN_dy_Omega' : r'Omega dN/dy',
                    'dN_dy_Xi' : r'Xi dN/dy',
                    'dET_deta' : r'Transverse energy [GeV]',
                    'mean_pT_pion' : r'Pion mean pT [GeV]',
                    'mean_pT_kaon' : r'Kaon mean pT [GeV]',
                    'mean_pT_proton' : r'Proton mean pT [GeV]',
                    'pT_fluct' : r'Mean pT fluctuations',
                    'v22' : v2_str,
                    'v32' : v3_str,
                    'v42' : v4_str,
}

system = 'Pb-Pb-2760'

#@st.cache(persist=True)
def load_design(system):
    #load the design
    design_file = SystemsInfo[system]["main_design_file"]
    range_file = SystemsInfo[system]["main_range_file"]
    design = pd.read_csv(design_file)
    design = design.drop("idx", axis=1)
    labels = design.keys()
    design_range = pd.read_csv(range_file)
    design_max = design_range['max'].values
    design_min = design_range['min'].values
    return design, labels, design_max, design_min


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_emu(system, idf):
    #load the emulator
    emu = dill.load(open('emulator/emulator-' + system + '-idf-' + str(idf) + '.dill', "rb"))
    return emu

@st.cache(persist=True)
def load_obs(system):
    observables = system_observables[system]
    nobs = len(observables)
    Yexp = Y_exp_data
    return observables, nobs, Yexp


#@st.cache(allow_output_mutation=True, show_spinner=False)
def emu_predict(emu, params):
    start = time.time()
    Yemu_cov = 0
    #Yemu_mean = emu.predict( np.array( [params] ), return_cov=False )
    Yemu_mean, Yemu_cov = emu.predict( np.array( [params] ), return_cov=True )
    end = time.time()
    time_emu = end - start
    return Yemu_mean, Yemu_cov, time_emu

#@st.cache(show_spinner=False)
def make_plot_altair(observables, Yemu_mean, Yemu_cov, Yexp, idf):
    for iobs, obs in enumerate(observables):
        xbins = np.array(obs_cent_list[system][obs])
        #centrality bins
        x = (xbins[:,0]+xbins[:,1])/2.
        #emulator prediction
        y_emu = Yemu_mean[obs][0]

        dy_emu = (np.diagonal(np.abs(Yemu_cov[obs, obs]))**.5)[:,0]
        df_emu = pd.DataFrame({'cent': x, 'yl':y_emu - dy_emu, "yh":y_emu + dy_emu})

        chart_emu = alt.Chart(df_emu).mark_area().encode(x='cent', y='yl', y2='yh').properties(width=150,height=150)

        #experiment
        exp_mean = Yexp[system][obs]['mean'][idf]
        exp_err = Yexp[system][obs]['err'][idf]
        df_exp = pd.DataFrame({"cent": x, obs:exp_mean, obs+"_dy":exp_err, obs+"_dy_low":exp_mean-exp_err, obs+"_dy_high":exp_mean+exp_err})

        # Adjust font size for the v_n's
        normal_font_size=14
        if (obs in ['v22','v32','v42']):
            normal_font_size=18

        pre_chart_exp=alt.Chart(df_exp)

        chart_exp = pre_chart_exp.mark_circle(color='Black').encode(
        x=alt.X( 'cent', axis=alt.Axis(title='Centrality (%)', titleFontSize=14), scale=alt.Scale(domain=(0, 70)) ),
        y=alt.Y(obs, axis=alt.Axis(title=obs_word_labels[obs], titleFontSize=normal_font_size), scale=alt.Scale(domain=(0, obs_lims[obs]))  )
        )

        # generate the error bars
        errorbars = pre_chart_exp.mark_errorbar().encode(
                x=alt.X('cent', axis=alt.Axis(title='')),
                y=alt.Y(obs+"_dy_low", axis=alt.Axis(title=''), scale=alt.Scale(domain=(0, obs_lims[obs]))  ),
                y2=alt.Y2(obs+"_dy_high"),
        )

        chart = alt.layer(chart_emu, chart_exp + errorbars)

        if iobs == 0:
            charts0 = chart
        if iobs in [1, 2]:
            charts0 = alt.hconcat(charts0, chart)

        if iobs == 3:
            charts1 = chart
        if iobs in [4, 5]:
            charts1 = alt.hconcat(charts1, chart)

        if iobs == 6:
            charts2 = chart
        if iobs in [7, 8]:
            charts2 = alt.hconcat(charts2, chart)

    charts0 = st.altair_chart(charts0)
    charts1 = st.altair_chart(charts1)
    charts2 = st.altair_chart(charts2)

    #return charts0, charts1, charts2

#@st.cache(suppress_st_warning=True)
def update_plot_altair(Yemu_mean, Yemu_cov, Yexp, idf, charts0, charts1, charts2):
    for iobs, obs in enumerate(observables):
        xbins = np.array(obs_cent_list[system][obs])
        #centrality bins
        x = (xbins[:,0]+xbins[:,1])/2.
        #emulator prediction
        y_emu = Yemu_mean[obs][0]

        dy_emu = (np.diagonal(np.abs(Yemu_cov[obs, obs]))**.5)[:,0]
        df_emu = pd.DataFrame({'cent': x, 'yl':y_emu - dy_emu, "yh":y_emu + dy_emu})

        charts0.add_rows(df_emu)


def make_plot_eta_zeta(params):
    T_low = 0.1
    T_high = 0.35
    T = np.linspace(T_low, T_high, 100)
    eta_s = eta_over_s(T, *params[7:11])
    zeta_s = zeta_over_s(T, *params[11:15])

    df_eta_zeta = pd.DataFrame({'T': T, 'eta':eta_s, 'zeta':zeta_s})

    chart_eta = alt.Chart(df_eta_zeta, title='Specific shear viscosity').mark_line(strokeWidth=4).encode(
    x=alt.X('T', axis=alt.Axis(title='T [GeV]', titleFontSize=14), scale=alt.Scale(domain=(T_low, T_high)) ),
    y=alt.Y('eta', axis=alt.Axis(title=eta_over_s_str, titleFontSize=14), scale=alt.Scale(domain=(0., 0.5 ))  ),
    color=alt.value("#FF0000")
    ).properties(width=150,height=150)

    chart_zeta = alt.Chart(df_eta_zeta, title='Specific bulk viscosity').mark_line(strokeWidth=4).encode(
    x=alt.X('T', axis=alt.Axis(title='T [GeV]', titleFontSize=14), scale=alt.Scale(domain=(T_low, T_high)) ),
    y=alt.Y('zeta', axis=alt.Axis(title=zeta_over_s_str, titleFontSize=14), scale=alt.Scale(domain=(0., 0.5 ))  ),
    color=alt.value("#FF0000")
    ).properties(width=150,height=150)

    #st_chart = st.altair_chart(chart)
    charts = alt.hconcat(chart_zeta, chart_eta)
    st.write(charts)

def main():
    st.title('Hadronic Observable Emulator for Heavy Ion Collisions')
    st.markdown('Our [model](https://inspirehep.net/literature/1821941) for the outcome of [ultrarelativistic heavy ion collisions](https://home.cern/science/physics/heavy-ions-and-quark-gluon-plasma) include many parameters which affects final hadronic observables in non-trivial ways. You can see how each observable (blue band) depends on the parameters by varying them using the sliders in the sidebar(left). All observables are plotted as a function of centrality for Pb nuclei collisions at'r'$\sqrt{s_{NN}} = 2.76$ TeV.')
    st.markdown('The experimentally measured observables by the [ALICE collaboration](https://home.cern/science/experiments/alice) are shown as black dots.')
    st.markdown('The last row displays the temperature dependence of the specific shear and bulk viscosities (red lines), as determined by different parameters on the left sidebar.')
    st.markdown('By default, these parameters are assigned the values that fit the experimental data *best* (maximize the likelihood).')
    st.markdown(r'An important modelling ingredient is the particlization model used to convert hydrodynamic fields into individual hadrons. Three different viscous correction models can be selected by clicking the "Particlization model" button below.')

    idf_names = ['Grad', 'Chapman-Enskog R.T.A', 'Pratt-Torrieri-Bernhard']
    idf_name = st.selectbox('Particlization model',idf_names)

    # Reset button
    st.markdown('<a href="javascript:window.location.href=window.location.href">Reset</a>', unsafe_allow_html=True)


    inverted_idf_label = dict([[v,k] for k,v in idf_label.items()])
    idf = inverted_idf_label[idf_name]

    #load the design
    design, labels, design_max, design_min = load_design(system)

    #load the emu
    emu = load_emu(system, idf)

    #load the exp obs
    observables, nobs, Yexp = load_obs(system)

    #initialize parameters
    params_0 = MAP_params[system][ idf_label_short[idf] ]
    params = []

    #updated params
    for i_s, s_name in enumerate(short_names.keys()):
        min = design_min[i_s]
        max = design_max[i_s]
        step = (max - min)/100.
        p = st.sidebar.slider(short_names[s_name], min_value=min, max_value=max, value=params_0[i_s], step=step)
        params.append(p)

    #get emu prediction
    Yemu_mean, Yemu_cov, time_emu = emu_predict(emu, params)

    #redraw plots
    make_plot_altair(observables, Yemu_mean, Yemu_cov, Yexp, idf)
    make_plot_eta_zeta(params)

    st.header('How it works')
    st.markdown('A description of the physics model and parameters can be found [here](https://indico.bnl.gov/event/6998/contributions/35770/attachments/27166/42261/JS_WS_2020_SIMS_v2.pdf).')
    st.markdown('The observables above (and additional ones not shown) are combined into [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) (PC).')
    st.markdown('A [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) (GP) is fitted to each of the dominant principal components by running our physics model on a coarse [space-filling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) set of points in parameter space.')
    st.markdown('The Gaussian Process is then able to interpolate between these points, while estimating its own uncertainty.')

    st.markdown('To update the widget with latest changes, click the button below, and then refresh your webpage.')
    if st.button('(Update widget)'):
        subprocess.run("git pull origin master", shell=True)


if __name__ == "__main__":
    main()
