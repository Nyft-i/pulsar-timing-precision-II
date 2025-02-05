import matplotlib.pyplot as plt
import numpy as np
import tim_sampling

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
  
def badness_at_point(sequence_type, args, glep, post_glitch_weight = 1.25):
  toas = toa_maker(sequence_type, args)[0]
  p_s = tim_sampling.find_sequence_period_info(sequence_type, args)[0]
  #print(p_s)
  
  badness = 0
  #print(toas)
  for toa in toas:
    dist1 = glep - toa
    dist2 = p_s - np.abs(glep - toa)
    if dist1>0:
      dist_left = dist1
      dist_right = post_glitch_weight * dist2
    else:
      dist_right = post_glitch_weight * np.abs(dist1)
      dist_left = dist2  
      
    dist = np.min((np.abs(dist_left), np.abs(dist_right)))
    dist_square = np.square(dist)
    
    badness += dist_square
  return badness
    
def goodness_graph(sequence_type, args, post_glitch_weight=1.25):
  fig = plt.figure(figsize=(8, 2))
  gs = fig.add_gridspec(1,2, width_ratios=[6, 1], wspace=0.01)
  axs = gs.subplots()
  
  toas = toa_maker(sequence_type, args)[0]
  #scale toas between 0 and 1
  a = 0
  b = toas.max()
  toas = (toas - a) / (b - a)
  
  p_s = tim_sampling.find_sequence_period_info(sequence_type, args)[0]
  x = np.linspace(0, p_s, 1000)
  y = np.zeros(len(x))
  y = np.array([badness_at_point(sequence_type, args, glep, post_glitch_weight) for glep in x])
  #normalise the values of y
  a = y.min()
  b = y.max()
  y = (y - a) / (b - a)
  G = 1-y  
  # normalise values of x
  a = x.min()
  b = x.max()
  x = (x - a) / (b - a)

  axs[0].plot(x, G, color='magenta', zorder= 10, linewidth=2)
  axs[0].eventplot(toas, linewidths=2, linelengths=np.max(y), lineoffsets=np.max(y)/2, colors = "black", alpha=0.3, label="observation", zorder=1)
  axs[0].legend(loc='lower right')
  
  # ylimit set to 1
  axs[0].set_ylim(0,1)
  axs[0].set_xlim(0,1)
  axs[0].set_title(sequence_type+" sequence 'goodness' against strategy phase")
  axs[0].set_xlabel("strategy phase")
  axs[0].set_ylabel("goodness score for glep \nlocation in strategy")
  
  textstr = '\n'.join((
    r'$k=%.4f$' % (args[3], ),
    r'$\Delta T_{min}=%.2f$d' % (args[0], ),
    r'$\Delta T_{max}=%.2f$d' % (args[2], ),
    r'$P_s=%.2f$d' % (p_s, ),
    'post-glitch favouring=%.2f' % (1/post_glitch_weight, )))
  
  axs[1].set_axis_off()
  axs[1].text(0, 0.5, textstr, fontsize=12, verticalalignment='center', horizontalalignment='left', transform=axs[1].transAxes)
  plt.savefig("goodness_plot_"+sequence_type+".png", dpi=400, bbox_inches="tight")

  


def main():
  post_glitch_weight = 2
  pgw = 1/post_glitch_weight
  goodness_graph('arithmetic', (0.5, 0, 15, 2.6019), pgw)
  
  plt.show()
if __name__ == "__main__":
  main()