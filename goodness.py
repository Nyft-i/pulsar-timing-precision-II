import matplotlib.pyplot as plt
import numpy as np
import tim_sampling
import pandas as pd

# explicit function to normalize array
def normalise(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)

def toa_maker(sequence_type, args):
  if sequence_type == 'logarithmic':
      cadence_start, marker_offset, max_gap, log_const = args
  elif sequence_type == 'arithmetic':
      cadence_start, marker_offset, max_gap, sequential_increase = args
  elif sequence_type == 'exponential':
      cadence_start, marker_offset, max_gap, exp_increase = args
  elif sequence_type == 'geometric':
      cadence_start, marker_offset, max_gap, multiplicative_increase = args
  elif sequence_type == 'periodic':
      cadence_start, marker_offset, max_gap, period = args
      cadence_start = period
  else:
      print("invalid sequence type. break.")
      
  start = marker_offset + cadence_start
  end = tim_sampling.find_sequence_period_info(sequence_type, args)[0]
  cadence = cadence_start

  all_obs = np.empty(0)
  endpoints = np.empty(0)

  marker = start
  while marker <= end:
      all_obs = np.append(all_obs, marker)
      #print("current cadence: ",cadence)

      if sequence_type=='logarithmic': cadence = (np.log(1/10 * cadence + 1) * log_const) 
      elif sequence_type=='arithmetic': cadence = cadence + sequential_increase
      elif sequence_type=='exponential': cadence = np.power(cadence,exp_increase)
      elif sequence_type=='geometric': cadence = cadence * multiplicative_increase
      elif sequence_type=='periodic': cadence = period 
      
      #write a not statement for the periodic case
      if(cadence > max_gap and (sequence_type!='periodic')):
          cadence = cadence_start
          endpoints = np.append(endpoints, marker)
      
      marker += cadence
      
  return all_obs, endpoints
    
def goodness_graph(sequence_type, args, y_param, true_val, csv_file, colour='orange'):


  p_s = tim_sampling.find_sequence_period_info(sequence_type, args)[0]
  
  #print(sequence_type, args)
  toas = toa_maker(sequence_type, args)[0]
  #print(toas)
  
  plt.xlim(0,p_s)
  
  # remove '_err' from y_param
  param_name = y_param[:-4]
  
  
  # load in a csv using pandas
  df = pd.read_csv(csv_file)
  # parse the filename to find the offset, which is the last number in the filename
  df['offset'] = df['filename'].str.extract(r'(\d+(\.\d+)?)(?!.*\d+(\.\d+)?)')[0]
  # set a refit flag if the filename contains the word 'refit'
  df['refit'] = df['filename'].str.contains('refit')
  # set a bayesian flag if the filename containts the word 'bay'
  df['bay'] = df['filename'].str.contains('bay')
  
  # seperates the refit rows
  # seperate the bayesian rows
  refit_df = df[df['refit'] == True]
  bay_df = df[df['bay'] == True]
  #print(len(refit_df))
  #print(len(bay_df))
  df = df[df['refit'] == False] 
  # drop the weird names from the regular dataframe
  df = df.drop(columns=['GLTD2_1', 'GLF0D2_1', 'GLTD2_1_err', 'GLF0D2_1_err']) 
  
  # print the column names of the bayesian dataframe
  #print(bay_df.columns)
  
  
  # delete the "GLTD_2" column in the bayesian dataframe IF and only if GLTD2_1 is present
  if 'GLTD2_1' in bay_df.columns:
    bay_df = bay_df.drop(columns=['GLTD_2'])
    bay_df = bay_df.drop(columns=['GLF0D_2'])
    bay_df = bay_df.drop(columns=['GLTD_2_err'])
    bay_df = bay_df.drop(columns=['GLF0D_2_err'])

    # rename the GLTD_1 column to GLTD_2, and GLTD2_1 to GLTD_1
    bay_df = bay_df.rename(columns={'GLTD_1': 'GLTD_2', 'GLTD2_1': 'GLTD_1', 'GLF0D_1': 'GLF0D_2', 'GLF0D2_1': 'GLF0D_1'})
    bay_df = bay_df.rename(columns={'GLTD_1_err': 'GLTD_2_err', 'GLTD2_1_err': 'GLTD_1_err', 'GLF0D_1_err': 'GLF0D_2_err', 'GLF0D2_1_err': 'GLF0D_1_err'})
    
    
  #print(bay_df)
  
  #normalise offset
  #convert offset to float
  offset_array = df['offset'].values.astype(float)
  refit_offset_array = refit_df['offset'].values.astype(float)
  bay_offset_array = bay_df['offset'].values.astype(float)
  a = 0
  b = p_s
  offset_array = b - (offset_array - a) * b / (b - a)
  refit_offset_array = b - (refit_offset_array - a) * b / (b - a)
  bay_offset_array = b - (bay_offset_array - a) * b / (b - a)
  
  found_val = df[y_param[:-4]].values
  refit_found_val = refit_df[y_param[:-4]].values
  bay_found_val = bay_df[y_param[:-4]].values
  original_error = df[y_param].values# relative error
  refit_original_error = refit_df[y_param].values
  bay_original_error = bay_df[y_param].values
  error_array = np.abs(original_error / true_val)
  refit_error_array = np.abs(refit_original_error / true_val)
  bay_error_array = np.abs(bay_original_error / true_val)
  
  # retrieve the indexes of all the elements in found_val where the value of GLTF0D_1 is 0, but only in the case of GLF0D_1 and GLTD_1 plots
  if y_param == 'GLF0D_2_err' or y_param == 'GLTD_2_err':
    zero_indexes = np.where(df['GLF0D_2'].values == 0)
    # remove these indexes from found_val, error_array and offset_array
    found_val = np.delete(found_val, zero_indexes)
    offset_array = np.delete(offset_array, zero_indexes)
    error_array = np.delete(error_array, zero_indexes)

  ret_sim = found_val / true_val -1
  refit_ret_sim = refit_found_val / true_val -1
  bay_ret_sim = bay_found_val / true_val -1
  
  # plot function
  if y_param == 'GLEP_1_err':
    x_vals = offset_array
    y_vals = found_val - true_val
    x_vals_refit = refit_offset_array
    y_vals_refit = refit_found_val - true_val
    x_vals_bay = bay_offset_array
    y_vals_bay = bay_found_val - true_val
    
  else:
    x_vals = offset_array
    y_vals = ret_sim
    x_vals_refit = refit_offset_array
    y_vals_refit = refit_ret_sim
    x_vals_bay = bay_offset_array
    y_vals_bay = bay_ret_sim
  
  #plot and plot refits
  plt.scatter(x_vals, y_vals, color=colour, zorder= 10, marker='x')
  plt.scatter(x_vals_refit, y_vals_refit, color='red', zorder= 10, marker='x')
  plt.scatter(x_vals_bay, y_vals_bay, color='red', zorder= 10, marker='x')
    
  #axis
  lowest_y, highest_y = plt.gca().get_ylim()
  print(lowest_y, highest_y)
  if np.nanmin(np.concatenate((y_vals, y_vals_refit, y_vals_bay, (0,)))) < lowest_y: lowest_y = np.nanmin(np.concatenate((y_vals, y_vals_refit, y_vals_bay, (0,))))
  if np.nanmax(np.concatenate((y_vals, y_vals_refit, y_vals_bay, (0,)))) > highest_y: highest_y = np.nanmax(np.concatenate((y_vals, y_vals_refit, y_vals_bay, (0,))))
  plt.ylim(lowest_y, highest_y)
  
  # plot the error bars after the above segment so that they do not influence the x and y lims
  plt.errorbar(x_vals, y_vals, xerr=0.25, yerr=error_array, fmt='x', ecolor=colour, zorder=5)
  plt.errorbar(x_vals_refit, y_vals_refit, xerr=0.25, yerr=refit_error_array, fmt='x', ecolor='red', zorder=5)
  plt.errorbar(x_vals_bay, y_vals_bay, xerr=0.25, yerr=bay_error_array, fmt='x', ecolor='red', zorder=5)
  
  # zero point line
  plt.axhline(y=0, color='magenta', linestyle='--')

  # toa plot
  plt.eventplot(toas, linewidths=2, linelengths=99999, colors = "black", alpha=0.3, label="observation", zorder=1)
  #plt.grid()
  
  return len(error_array)
  
def plot_28():
  # create figure and gs
  fig = plt.figure(figsize=(20, 10))
  gs = fig.add_gridspec(8,5, wspace=0.1, hspace=0.25)
  axs = gs.subplots(sharex='col', sharey='row')
  
  #pulsar properties
  T_GLEP_1 = 60000
  T_GLF0_1 = 3e-6
  T_GLF1_1 = -1.5e-14
  T_GLF0D_1 = 3e-7
  T_GLTD_1 = 100
  T_GLF0D_2 = 1.5e-7
  T_GLTD_2 = 7
  
  
  plt.suptitle("relative deviation of found parameter values to true values.", y=1)
  
  #geo
  strategy = 'periodic'
  args = (0.5, 0, 25, 7)
  file = '.\\par_reader\\seed_123\\7d\\peri.csv'
  per_col = "limegreen"
  

  
  plt.sca(axs[1,0])
  yparam = "GLEP_1_err"
  true_val = T_GLEP_1
  plt.ylabel(r"$t_g-t_{g,true}$")
  goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
  plt.sca(axs[2,0])
  yparam = "GLF0_1_err"
  true_val = T_GLF0_1
  plt.ylabel(r"$\Delta \nu/\Delta \nu_{true} - 1$")
  goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
  plt.sca(axs[3,0])
  yparam = "GLF1_1_err"
  true_val = T_GLF1_1
  plt.ylabel(r"$\dot{\nu}/\dot{\nu}_{true} - 1$")
  goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
  plt.sca(axs[4,0])
  yparam = "GLF0D_1_err"
  true_val = T_GLF0D_1
  plt.ylabel(r"$\Delta \nu_d/\Delta \nu_{d,true} - 1$")
  goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
  plt.sca(axs[5,0])
  yparam = "GLTD_1_err"
  true_val = T_GLTD_1
  plt.ylabel(r"$\tau_d/\tau_{d,true} - 1$")
  num_double = goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
  plt.sca(axs[6,0])
  yparam = "GLF0D_2_err"
  true_val = T_GLF0D_2
  plt.ylabel(r"$\Delta \nu_{d,2}/\Delta \nu_{d,2,true} - 1$")
  goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
  plt.sca(axs[7,0])
  yparam = "GLTD_2_err"
  true_val = T_GLTD_2
  plt.ylabel(r"$\tau_{d,2}/\tau_{d,2,true} - 1$")
  num_single = goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
  plt.xlabel("days")
  
 
  #peri information
  plt.sca(axs[0,0])
  # hide axis
  plt.axis('off')
  textstr = '\n'.join((
    r'periodic sequence',
    r'$\Delta T = 7$',
    r'single/double exp. found: %d/%d' % (num_single, num_double)))
  plt.text(0.5, 1, textstr, fontsize=8, verticalalignment='top', horizontalalignment='center', transform=plt.gca().transAxes)

  #geometric
  strategy = 'geometric'
  args = (0.5, 0, 25, 1.7391)
  file = '.\\par_reader\\seed_123\\7d\\geo.csv'
  per_col = "orange"
  
 
  plt.sca(axs[1,1])
  goodness_graph(strategy, args, "GLEP_1_err", T_GLEP_1, file, colour=per_col)
  plt.sca(axs[2,1])
  goodness_graph(strategy, args, "GLF0_1_err", T_GLF0_1, file, colour=per_col)
  plt.sca(axs[3,1])
  goodness_graph(strategy, args, "GLF1_1_err", T_GLF1_1, file, colour=per_col)
  plt.sca(axs[4,1])
  goodness_graph(strategy, args, "GLF0D_1_err", T_GLF0D_1, file, colour=per_col)
  plt.sca(axs[5,1])
  num_double = goodness_graph(strategy, args, "GLTD_1_err", T_GLTD_1, file, colour=per_col)
  plt.sca(axs[6,1])
  goodness_graph(strategy, args, "GLF0D_2_err", T_GLF0D_2, file, colour=per_col)
  plt.sca(axs[7,1])
  num_single = goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)
  plt.xlabel("days")

  #geo information
  plt.sca(axs[0,1])
  # hide axis
  plt.axis('off')
  textstr = '\n'.join((
    r'geometric sequence',
    r'$\Delta T_{new} = k_g \Delta T$',
    r'$k_g = 1.7391$',
    r'$\Delta T_{max} = 25$',
    r'$\Delta T_{min} = 0.5$',
    r'single/double exp. found: %d/%d' % (num_single, num_double)))
  plt.text(0.5, 1, textstr, fontsize=8, verticalalignment='top', horizontalalignment='center', transform=plt.gca().transAxes)
  
  #arithmetic
  strategy = 'arithmetic'
  args = (0.5, 0, 15, 2.6019)
  file = '.\\par_reader\\seed_123\\7d\\arith.csv'
  per_col = "tab:blue"
  
  
  plt.sca(axs[1,2])
  goodness_graph(strategy, args, "GLEP_1_err", T_GLEP_1, file, colour=per_col)
  plt.sca(axs[2,2])
  goodness_graph(strategy, args, "GLF0_1_err", T_GLF0_1, file, colour=per_col)
  plt.sca(axs[3,2])
  goodness_graph(strategy, args, "GLF1_1_err", T_GLF1_1, file, colour=per_col)
  plt.sca(axs[4,2])
  goodness_graph(strategy, args, "GLF0D_1_err", T_GLF0D_1, file, colour=per_col)
  plt.sca(axs[5,2])
  num_double = goodness_graph(strategy, args, "GLTD_1_err", T_GLTD_1, file, colour=per_col)
  plt.sca(axs[6,2])
  goodness_graph(strategy, args, "GLF0D_2_err", T_GLF0D_2, file, colour=per_col)
  plt.sca(axs[7,2])
  num_single = goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)
  plt.xlabel("days")

  #arith information
  plt.sca(axs[0,2])
  # hide axis
  plt.axis('off')
  textstr = '\n'.join((
    r'arithmetic sequence',
    r'$\Delta T_{new} = k_a + \Delta T$',
    r'$k_a = 2.6019$',
    r'$\Delta T_{max} = 15$',
    r'$\Delta T_{min} = 0.5$',
    r'single/double exp. found: %d/%d' % (num_single, num_double)))
  plt.text(0.5, 1, textstr, fontsize=8, verticalalignment='top', horizontalalignment='center', transform=plt.gca().transAxes)
  
  # logarithmic
  strategy = 'logarithmic'
  args = (0.5, 0, 20, 24.7678)
  file = '.\\par_reader\\seed_123\\7d\\log.csv'
  per_col = "mediumorchid"
  
  
  plt.sca(axs[1,3])
  goodness_graph(strategy, args, "GLEP_1_err", T_GLEP_1, file, colour=per_col)
  plt.sca(axs[2,3])
  goodness_graph(strategy, args, "GLF0_1_err", T_GLF0_1, file, colour=per_col)
  plt.sca(axs[3,3])
  goodness_graph(strategy, args, "GLF1_1_err", T_GLF1_1, file, colour=per_col)
  plt.sca(axs[4,3])
  goodness_graph(strategy, args, "GLF0D_1_err", T_GLF0D_1, file, colour=per_col)
  plt.sca(axs[5,3])
  num_double = goodness_graph(strategy, args, "GLTD_1_err", T_GLTD_1, file, colour=per_col)
  plt.sca(axs[6,3])
  goodness_graph(strategy, args, "GLF0D_2_err", T_GLF0D_2, file, colour=per_col)
  plt.sca(axs[7,3])
  num_single = goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)
  plt.xlabel("days")
  
  #log information
  plt.sca(axs[0,3])
  # hide axis
  plt.axis('off')
  textstr = '\n'.join((
    r'logarithmic sequence',
    r'$\Delta T_{new} = k_l \ln \left( \frac{\Delta T}{10}+1 \right)$',
    r'$k_l = 24.7678$',
    r'$\Delta T_{max} = 20$',
    r'$\Delta T_{min} = 0.5$',
    r'single/double exp. found: %d/%d' % (num_single, num_double)))
  plt.text(0.5, 1, textstr, fontsize=8, verticalalignment='top', horizontalalignment='center', transform=plt.gca().transAxes)
  
  
  # hide top right component
  plt.sca(axs[0,4])
  plt.axis('off')
  
  # true value column
  plt.sca(axs[1,4])
  plt.axis('off')
  yparam = "GLEP_1_err"
  true_val = T_GLEP_1
  text_str = r"True %s: %.2e" % (yparam[:-4], true_val, )
  plt.text(0.05, 0.5, text_str, fontsize=10, verticalalignment='center', horizontalalignment='left', transform=plt.gca().transAxes)
  
  plt.sca(axs[2,4])
  plt.axis('off')
  yparam = "GLF0_1_err"
  true_val = T_GLF0_1
  text_str = r"True %s: %.2e" % (yparam[:-4], true_val, )
  plt.text(0.05, 0.5, text_str, fontsize=10, verticalalignment='center', horizontalalignment='left', transform=plt.gca().transAxes)
  
  plt.sca(axs[3,4])
  plt.axis('off')
  yparam = "GLF1_1_err"
  true_val = T_GLF1_1
  text_str = r"True %s: %.2e" % (yparam[:-4], true_val, )
  plt.text(0.05, 0.5, text_str, fontsize=10, verticalalignment='center', horizontalalignment='left', transform=plt.gca().transAxes)
  
  plt.sca(axs[4,4])
  plt.axis('off')
  yparam = "GLF0D_1_err"
  true_val = T_GLF0D_1
  text_str = r"True %s: %.2e" % (yparam[:-4], true_val, )
  plt.text(0.05, 0.5, text_str, fontsize=10, verticalalignment='center', horizontalalignment='left', transform=plt.gca().transAxes)
  
  plt.sca(axs[5,4])
  plt.axis('off')
  yparam = "GLTD_1_err"
  true_val = T_GLTD_1
  text_str = r"True %s: %.2e" % (yparam[:-4], true_val, )
  plt.text(0.05, 0.5, text_str, fontsize=10, verticalalignment='center', horizontalalignment='left', transform=plt.gca().transAxes)
  
  plt.sca(axs[6,4])
  plt.axis('off')
  yparam = "GLF0D_2_err"
  true_val = T_GLF0D_2
  text_str = r"True %s: %.2e" % (yparam[:-4], true_val, )
  plt.text(0.05, 0.5, text_str, fontsize=10, verticalalignment='center', horizontalalignment='left', transform=plt.gca().transAxes)
  
  plt.sca(axs[7,4])
  plt.axis('off')
  yparam = "GLTD_2_err"
  true_val = T_GLTD_2
  text_str = r"True %s: %.2e" % (yparam[:-4], true_val, )
  plt.text(0.05, 0.5, text_str, fontsize=10, verticalalignment='center', horizontalalignment='left', transform=plt.gca().transAxes)
  
  plt.savefig("stn_123_bayes.png", dpi=400, bbox_inches="tight")
  
  plt.show()
    
def plot_results():
  # create figure and gs
  fig = plt.figure(figsize=(15, 3))
  gs = fig.add_gridspec(2, 4, wspace=0.1, hspace=0.25)
  axs = gs.subplots(sharex='col', sharey='row')
  
  #pulsar properties
  T_GLEP_1 = 60000
  T_GLF0_1 = 3e-6
  T_GLF1_1 = -1.5e-14
  T_GLF0D_1 = 3e-7
  T_GLTD_1 = 100
  T_GLF0D_2 = 1.5e-7
  T_GLTD_2 = 7
  
  
  plt.suptitle(r"$\tau_{d,short} = 7$d", y=1.05, fontsize=20)
  
  # arith
  strategy = 'arithmetic'
  args = (0.5, 0, 15, 2.6019)
  per_col = "tab:blue"
  
  # case (a)
  file = '.\\par_reader\\seed_123\\5d\\arith.csv'
  plt.sca(axs[0,0])
  yparam = "GLTD_2_err"
  true_val = T_GLTD_2
  plt.title("arithmetic", fontsize=12)
  plt.ylabel("case (a) \n" r"$\frac{\tau_{d,short}}{\tau_{d,short,true}} - 1$", fontsize=15)
  goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
  
  # case (b)
  file = '.\\par_reader\\seed_123\\5d\\arith.csv'
  plt.sca(axs[1,0])
  yparam = "GLTD_2_err"
  true_val = T_GLTD_2
  plt.ylabel("case (b) \n" r"$\frac{\tau_{d,short}}{\tau_{d,short,true}} - 1$", fontsize=15)
  goodness_graph(strategy, args, yparam, true_val, file, colour=per_col)
 
  plt.xlabel("days")
 
  # geometric
  strategy = 'geometric'
  args = (0.5, 0, 25, 1.7391)
  per_col = "orange"
  
  # case (a)
  file = '.\\par_reader\\seed_123\\5d\\geo.csv'
  plt.sca(axs[0,1])
  plt.title("geometric", fontsize=12)
  goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)
  
  # case (b)
  file = '.\\par_reader\\seed_123\\5d\\geo.csv'
  plt.sca(axs[1,1])
  goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)
  
  plt.xlabel("days")
  
  # logarithmic
  strategy = 'logarithmic'
  args = (0.5, 0, 20, 24.7678)
  per_col = "mediumorchid"
  
  # case (a)
  file = '.\\par_reader\\seed_123\\5d\\log.csv'
  plt.sca(axs[0,2])
  plt.title("logarithmic", fontsize=12)
  goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)
  
  # case (b)
  file = '.\\par_reader\\seed_123\\5d\\log.csv'
  plt.sca(axs[1,2])
  goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)
  
  plt.xlabel("days")
  
  # periodic
  strategy = 'periodic'
  args = (0.5, 0, 25, 7)
  per_col = "limegreen"
  
  # case (a)
  file = '.\\par_reader\\seed_123\\5d\\peri.csv'
  plt.sca(axs[0,3])
  plt.title("periodic", fontsize=12)
  goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)
  
  # case (b)
  file = '.\\par_reader\\seed_123\\5d\\peri.csv'
  plt.sca(axs[1,3])
  goodness_graph(strategy, args, "GLTD_2_err", T_GLTD_2, file, colour=per_col)

  plt.xlabel("days")


  
  plt.savefig("results_test.png", dpi=400, bbox_inches="tight")
  
  plt.show()

  
def main():
  plot_28()
  
  plt.show()

if __name__ == "__main__":
  main()