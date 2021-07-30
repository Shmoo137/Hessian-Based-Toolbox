import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as ticker
from decimal import Decimal

from utility_general import replace_character_in_string

def add_shade_for_training_region(ax, fig_details, color):
    if fig_details.find('much_smaller_transition') != -1:
        ax.axvspan(0.8, 3, facecolor=color, zorder=0, lw=0)
    elif fig_details.find('smaller_transition') != -1:
        ax.axvspan(0.75, 10, facecolor=color, zorder=0, lw=0)
    elif fig_details.find('transition') != -1:
        ax.axvspan(0.75, 20, facecolor=color, zorder=0, lw=0)
    elif fig_details.find('plateaus') != -1:
        # Left plateau
        left_letter = fig_details[fig_details.find('plateaus') + 9]
        right_letter = fig_details[fig_details.find('plateaus') + 10]
        if left_letter == 'a':
            ax.axvspan(0, 0.25, facecolor=color, zorder=0, lw=0)
        elif left_letter == 'b':
            ax.axvspan(0.15, 0.4, facecolor=color, zorder=0, lw=0)
        else:
            print("Unknown first part of plateaus dataset!")
        
        # Right plateaus
        if right_letter == 'a':
            ax.axvspan(36, 40, facecolor=color, zorder=0, lw=0) # OP 0.98-1
        elif right_letter == 'b':
            ax.axvspan(30, 40, facecolor=color, zorder=0, lw=0) # OP 0.88-1
        elif right_letter == 'c':
            ax.axvspan(28, 40, facecolor=color, zorder=0, lw=0) # OP 0.80-1
        elif right_letter == 'd':
            ax.axvspan(28, 30, facecolor=color, zorder=0, lw=0) # OP 0.8-0.88
        elif right_letter == 'e':
            ax.axvspan(26, 28, facecolor=color, zorder=0, lw=0) # OP 0.66-0.8
        elif right_letter == 'f':
            ax.axvspan(20, 22, facecolor=color, zorder=0, lw=0) # OP 0.24-0.37
        elif right_letter == 'g':
            ax.axvspan(10, 12, facecolor=color, zorder=0, lw=0) # OP 0.008-0.02
        elif right_letter == 'h':
            ax.axvspan(4, 6, facecolor=color, zorder=0, lw=0) # OP 0.0001-0.0007
        else:
            print("Unknown first part of plateaus dataset!")
    else:
        ax.axvspan(0, 40, facecolor=color, zorder=0, lw=0)

def make_IF_analysis(chosen_test_examples, mask, folder_influence, file_name, U_array, U_testarray, test_order_parameters, misclassified_array, fig_details, folder_figure, model_accuracy):

    fig_title = replace_character_in_string(fig_details, '_', ' ') + ', AC=' + str(100 * model_accuracy) + '%'
    antimask = np.argsort(mask)

    def testV1toOP(x):
        return np.interp(x, np.concatenate((U_testarray, np.array([100]))), np.concatenate((test_order_parameters, np.array([1.0]))))

    def OPtotestV1(x):
        return np.interp(x, np.concatenate((test_order_parameters, np.array([1.0]))), np.concatenate((U_testarray, np.array([100]))))

    for test_sample in chosen_test_examples:

        # Influence functions of all train elements for one test example
        with open(folder_influence + '/' + file_name + '_test' + str(test_sample) + '.txt') as filelabels:
            influence_functions = np.loadtxt(filelabels, dtype=float)

        antimasked_inf_funs = influence_functions[antimask]
        sorting_indices = np.argsort(antimasked_inf_funs)

        min_y = 5 * np.min(influence_functions)
        if min_y > -1e-4:
            min_y = -9e-3
        max_y = 5 * np.max(influence_functions)
        if max_y < 1e-4:
            max_y = 9e-3

        U_test_value = U_testarray[test_sample]

        if test_sample == 16:
            print("Positive test point")
            print("Most helpful points: ", sorting_indices[-5:])
            print("Most harmful points: ", sorting_indices[:5])

        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(U_array, antimasked_inf_funs, 'o')
        ax.plot(U_array[sorting_indices[:5]], antimasked_inf_funs[sorting_indices[:5]], 'ro')
        ax.plot(U_array[sorting_indices[-5:]], antimasked_inf_funs[sorting_indices[-5:]], 'go')

        #ax.plot(U_array, training_U, linestyle='dotted')

        if test_sample in misclassified_array:
            ax.text(5, 1e-7, 'misclassified')

        xticks_location = np.concatenate((np.array([0, 0.5, 1]), np.array([2,10,20,40]))) #,55,80,140
        xticks_labels = np.concatenate((np.array(['0', '0.5', '1']), np.array(['2','10','20','40']))) #,'55','80','140'

        ax.plot([U_test_value,U_test_value], [min_y, max_y])
        ax.set_ylabel('Influence function value')
        ax.set_xlabel('$V_1 /\, J$ for which training example was calculated')
        #plt.tight_layout()
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.set_xscale('symlog', linthresh=1.0)
        ax.set_xticks(xticks_location)
        ax.set_xticklabels(xticks_labels)
        #plt.yticks(yticks_location)
        ax.set_ylim(min_y, max_y)
        ax.set_title(fig_title)

        secax = ax.secondary_xaxis('top', xlabel='order parameter')#, functions=(testV1toOP, OPtotestV1) functions=(symlogtestV1toOP, symlogOPtotestV1))
        secax.set_xscale('symlog', linthresh=1.0)
        secax.set_xticks(OPtotestV1([1e-6, 1e-5, 1e-4, 1e-2, 0.1, 0.9]))
        secax.set_xticklabels([1e-6, 1e-5, 1e-4, 1e-2, 0.1, 0.9])
        if U_testarray[-1] <= np.ceil(U_array[-1]):
            ax.set_xlim(0, np.ceil(U_array[-1]))
        else:
            ax.set_xlim(0, np.ceil(U_testarray[-1]))
        xticks_location = np.concatenate((np.array([0, 0.5, 1]), np.array([2,10,20,40])))
        xticks_labels = np.concatenate((np.array(['0', '0.5', '1']), np.array(['2','10','20','40'])))
        ax.set_xticks(xticks_location)
        ax.set_xticklabels(xticks_labels)

        ax.plot([1,1], [min_y, max_y], linestyle='dashed')

        plt.savefig('./' + folder_figure + '/' + file_name + '_test' + str(test_sample) + '.png')
        plt.close()

def BOOPtotestV1(x, U_testarray):
    test_order_parameters = np.array([0.0007924, 0.0008217, 0.00084876, 0.00087378, 0.00089702, 0.0009188, 0.00093945, 0.00095933, 0.00097879, 0.00099819, 0.00099894, 0.00103763, 0.00108013, 0.00112926, 0.00118839, 0.00126199, 0.00135649, 0.00148159, 0.00165183, 0.00188789, 0.00221549, 0.00266045, 0.00324345, 0.00398106, 0.00488938, 0.00598399, 0.00727859, 0.00878493, 0.01051331, 0.01247299, 0.01251256, 0.01591151, 0.02014178, 0.02535248, 0.03168972, 0.03926089, 0.04806471, 0.05787732, 0.06811746, 0.07779068, 0.08567459, 0.09077531, 0.09275464, 0.09196686, 0.08915504, 0.08512111, 0.08052173, 0.0757901, 0.07116098, 0.06674097])
    return np.interp(x, np.concatenate((test_order_parameters[:-7], np.array([0.093]))), np.concatenate((U_testarray[:-7], np.array([100]))))

def make_LEES_plot(various_lees, ms, test_loss, rue, chosen_test_examples, U_array, U_testarray, test_order_parameters, fig_details, folder_figure, model_accuracy):

    fig_title = replace_character_in_string(fig_details, '_', ' ') + ', AC=' + str(100 * model_accuracy) + '\%'

    no_tresholds = len(ms)
    test_set_size = len(chosen_test_examples)
    various_lees = various_lees.reshape((no_tresholds, test_set_size))

    # Overriding fonts
    plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{cmbright}"})

    plt.rc('text', usetex=True)

    # Seaborn style set
    sns.set(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': 'dashed', "grid.color": "0.6", 'axes.edgecolor': '.1'})

    fig, ax = plt.subplots(constrained_layout=True)
    
    add_shade_for_training_region(ax, fig_details, 'lightgray')

    for i in range(no_tresholds):
        ax.plot(U_testarray[chosen_test_examples], various_lees[i], label = str(ms[i]))

    ax.plot(U_testarray[chosen_test_examples], test_loss.detach().numpy()[chosen_test_examples], label = 'Test loss')
    ax.errorbar(U_testarray[chosen_test_examples], np.zeros(len(chosen_test_examples)), yerr=rue, label = 'RUE')
    ax.legend()

    #for i in range(len(chosen_test_examples)):
    #    if lees[i] > 0.1:
    #        ax.annotate("{:.2f}".format(U_testarray[i]) + "-" + "{:.1E}".format(Decimal(test_order_parameters[i])), (U_testarray[i], lees[i]), size=5)

    ax.set_xscale('symlog', linthresh=1.0)
    xticks_location = np.concatenate((np.array([0, 0.5, 1]), np.array([2,10,20,40]))) #,55,80,140
    xticks_labels = np.concatenate((np.array(['0', '0.5', '1']), np.array(['2','10','20','40']))) #,'55','80','140'
    ax.set_xticks(xticks_location)
    ax.set_xticklabels(xticks_labels)
    ax.set_xlabel('$V_1/\,J$ ', family="serif")
    ax.set_ylabel('Local ensemble extrapolation score', family="serif")
    ax.set_title(fig_title, family="serif")

    def testV1toOP(x):
        return np.interp(x, np.concatenate((U_testarray[chosen_test_examples], np.array([100]))), np.concatenate((test_order_parameters[chosen_test_examples], np.array([1.0]))))
    
    def OPtotestV1(x):
        return np.interp(x, np.concatenate((test_order_parameters[chosen_test_examples], np.array([1.0]))), np.concatenate((U_testarray[chosen_test_examples], np.array([100]))))

    min_y, max_y = ax.get_ylim()

    secax = ax.secondary_xaxis('top', xlabel='order parameter')#, functions=(testV1toOP, OPtotestV1) functions=(symlogtestV1toOP, symlogOPtotestV1))
    secax.set_xscale('symlog', linthresh=1.0)#, linthresh=testV1toOP(1.0))
    secax.set_xticks(OPtotestV1([1e-6, 1e-4, 1e-2, 0.1, 0.9]))
    secax.set_xticklabels([1e-6, 1e-4, 1e-2, 0.1, 0.9])
    
    if U_testarray[-1] <= np.ceil(U_array[-1]):
        ax.set_xlim(0, np.ceil(U_array[-1]))
    else:
        ax.set_xlim(0, np.ceil(U_testarray[-1]))
    xticks_location = np.concatenate((np.array([0, 0.5, 1]), np.array([2,10,20,40])))
    xticks_labels = np.concatenate((np.array(['0', '0.5', '1']), np.array(['2','10','20','40'])))
    ax.set_xticks(xticks_location)
    ax.set_xticklabels(xticks_labels)

    ax.plot([1,1], [min_y, max_y], linestyle='dashed')

    plt.savefig('./' + folder_figure + '/extrapolation_' + fig_details + '.pdf', bbox_inches='tight')

def make_RUE_plot(rue, test_loss, chosen_test_examples, U_array, U_testarray, test_order_parameters, fig_details, folder_figure, model_accuracy, WITH_TEST_LOSS = False, LOSS_PERCENTAGE=True):
    
    test_loss = test_loss.detach().numpy()
    fig_title = replace_character_in_string(fig_details, '_', ' ')  + ', AC=' + str(100 * model_accuracy) + '\%'

    # Overriding fonts
    plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{cmbright}"})
    #plt.rc('font',**{'family':'serif','serif':['Times']})
    plt.rc('text', usetex=True)

    # Seaborn style set 
    sns.set(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': 'dashed', "grid.color": "0.6", 'axes.edgecolor': '.1'})

    if LOSS_PERCENTAGE is True:
        rue_perc = np.divide(rue, test_loss[chosen_test_examples])
    else:
        rue_perc = rue

    fig, ax = plt.subplots(constrained_layout=True)
    
    add_shade_for_training_region(ax, fig_details, 'lightgray')

    if WITH_TEST_LOSS is True:
        ax.errorbar(U_testarray[chosen_test_examples], test_loss[chosen_test_examples], yerr=rue_perc, ls='none', marker='o', markersize=2)
    else:
        ax.errorbar(U_testarray[chosen_test_examples], np.zeros(50)[chosen_test_examples], yerr=rue_perc, ls='none', marker='o', markersize=2)

    for i in range(len(chosen_test_examples)):
        if rue_perc[i] > 0.2:
            ax.annotate("{:.2f}".format(U_testarray[i]) + "-" + "{:.1E}".format(Decimal(test_order_parameters[i])), (U_testarray[i], test_loss[i]), size=5)

    ax.set_xscale('symlog', linthresh=1.0)
    xticks_location = np.concatenate((np.array([0, 0.5, 1]), np.array([2,10,20,40]))) #,55,80,140
    xticks_labels = np.concatenate((np.array(['0', '0.5', '1']), np.array(['2','10','20','40']))) #,'55','80','140'
    ax.set_xticks(xticks_location)
    ax.set_xticklabels(xticks_labels)
    ax.set_xlabel('$V_1/\,J$ ', family="serif")
    ax.set_ylabel('Test loss', family="serif")
    ax.set_title(fig_title, family="serif")

    def testV1toOP(x):
        return np.interp(x, np.concatenate((U_testarray[chosen_test_examples], np.array([100]))), np.concatenate((test_order_parameters[chosen_test_examples], np.array([1.0]))))
    
    def OPtotestV1(x):
        return np.interp(x, np.concatenate((test_order_parameters[chosen_test_examples], np.array([1.0]))), np.concatenate((U_testarray[chosen_test_examples], np.array([100]))))

    min_y, max_y = ax.get_ylim()

    secax = ax.secondary_xaxis('top', xlabel='order parameter')#, functions=(testV1toOP, OPtotestV1) functions=(symlogtestV1toOP, symlogOPtotestV1))
    secax.set_xscale('symlog', linthresh=1.0)#, linthresh=testV1toOP(1.0))
    secax.set_xticks(OPtotestV1([1e-6, 1e-4, 1e-2, 0.1, 0.9]))
    secax.set_xticklabels([1e-6, 1e-4, 1e-2, 0.1, 0.9])
    
    if U_testarray[-1] <= np.ceil(U_array[-1]):
        ax.set_xlim(0, np.ceil(U_array[-1]))
    else:
        ax.set_xlim(0, np.ceil(U_testarray[-1]))
    xticks_location = np.concatenate((np.array([0, 0.5, 1]), np.array([2,10,20,40])))
    xticks_labels = np.concatenate((np.array(['0', '0.5', '1']), np.array(['2','10','20','40'])))
    ax.set_xticks(xticks_location)
    ax.set_xticklabels(xticks_labels)

    ax.plot([1,1], [min_y, max_y], linestyle='dashed')

    if WITH_TEST_LOSS is True and LOSS_PERCENTAGE is True:
        plt.savefig('./' + folder_figure + '/error_bars_' + fig_details + '.pdf', bbox_inches='tight')
    elif WITH_TEST_LOSS is True and LOSS_PERCENTAGE is False:
        plt.savefig('./' + folder_figure + '/error_bars_' + fig_details + '_notpercent.pdf', bbox_inches='tight')
    elif WITH_TEST_LOSS is False and LOSS_PERCENTAGE is True:
        plt.savefig('./' + folder_figure + '/error_vanilla_bars_' + fig_details + '.pdf', bbox_inches='tight')
    else:
        plt.savefig('./' + folder_figure + '/error_vanilla_bars_' + fig_details + '_notpercent.pdf', bbox_inches='tight')

def make_neurons_outputs_plot(neurons, chosen_test_examples, U_array, U_testarray, test_order_parameters, fig_details, folder_figure, model_accuracy):

    fig_title = replace_character_in_string(fig_details, '_', ' ') + ', AC=' + str(100 * model_accuracy) + '\%'

    neuron0 = neurons[0]
    neuron1 = neurons[1]

    if fig_details.find('LL_BO_CDWII') != -1:
        neuron2 = neurons[2]

    # Overriding fonts
    plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{cmbright}"})
    #plt.rc('font',**{'family':'serif','serif':['Times']})
    plt.rc('text', usetex=True)

    # Seaborn style set 
    sns.set(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': 'dashed', "grid.color": "0.6", 'axes.edgecolor': '.1'})

    fig, ax = plt.subplots(constrained_layout=True)
    
    add_shade_for_training_region(ax, fig_details, 'lightgray')

    ax.plot(U_testarray[chosen_test_examples], neuron0[chosen_test_examples], label='Neuron 0')
    ax.plot(U_testarray[chosen_test_examples], neuron1[chosen_test_examples], label='Neuron 1')
    if fig_details.find('LL_BO_CDWII') != -1:
        ax.plot(U_testarray[chosen_test_examples], neuron2[chosen_test_examples], label='Neuron 2')
    ax.legend()

    ax.set_xscale('symlog', linthresh=1.0)
    ax.set_xlabel('$V_1/\,J$ ', family="serif")
    if fig_details.find('LL_BO_CDWII') != -1:
        ax.set_ylabel("Last three neurons' outputs", family="serif")
    else:
        ax.set_ylabel("Last two neurons' outputs", family="serif")
    ax.set_title(fig_title, family="serif")

    def testV1toOP(x):
        return np.interp(x, np.concatenate((U_testarray[chosen_test_examples], np.array([100]))), np.concatenate((test_order_parameters[chosen_test_examples], np.array([1.0]))))
    
    def OPtotestV1(x):
        return np.interp(x, np.concatenate((test_order_parameters[chosen_test_examples], np.array([1.0]))), np.concatenate((U_testarray[chosen_test_examples], np.array([100]))))

    min_y, max_y = ax.get_ylim()

    secax = ax.secondary_xaxis('top', xlabel='order parameter')#, functions=(testV1toOP, OPtotestV1) functions=(symlogtestV1toOP, symlogOPtotestV1))
    secax.set_xscale('symlog', linthresh=1.0)#, linthresh=testV1toOP(1.0))
    secax.set_xticks(OPtotestV1([1e-6, 1e-4, 1e-2, 0.1, 0.9]))
    secax.set_xticklabels([1e-6, 1e-4, 1e-2, 0.1, 0.9])

    if U_testarray[-1] <= np.ceil(U_array[-1]):
        ax.set_xlim(0, np.ceil(U_array[-1]))
    else:
        ax.set_xlim(0, np.ceil(U_testarray[-1]))
    xticks_location = np.concatenate((np.array([0, 0.5, 1]), np.array([2,10,20,40])))
    xticks_labels = np.concatenate((np.array(['0', '0.5', '1']), np.array(['2','10','20','40'])))
    ax.set_xticks(xticks_location)
    ax.set_xticklabels(xticks_labels)

    ax.plot([1,1], [min_y, max_y], linestyle='dashed')

    plt.savefig('./' + folder_figure + '/neurons_outputs_' + fig_details + '.pdf', bbox_inches='tight')