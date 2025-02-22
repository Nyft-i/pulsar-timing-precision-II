# program which reads in all the data from a par file and puts it into a pandas dataframe
import pandas as pd
import os
import argparse

def dict_creator(f):
  par_cols = ["name", "value", "fitting", "error"]

  print("Reading in file: " + f)
    
  # read in the file
  df = pd.read_csv(f, sep='\s+', header=None, names=par_cols, float_precision='round_trip')
  df = df.T
  df.columns = df.iloc[0]
  df.insert(0,'filename',f)
  #print(df)

  # use only the data
  dict = df.to_dict(orient='records')[1]
  return dict  

def dict_from_txt(f):
  par_cols = ["name", "value", "error"]

  print("Reading in file: " + f)
    
  # read in the file
  df = pd.read_csv(f, sep='\s+', header=None, names=par_cols, float_precision='round_trip')
  print(df)
  df = df.T
  df.columns = df.iloc[0]
  df.insert(0,'filename',f)
  print(df)

  # use only the data
  dict = df.to_dict(orient='records')[1]
  return dict  

# main function
def main():
  
  dicts = {}
  
  if recursive:
    curr_index = 0
    for subdir, dirs, files in os.walk(directory):
      for file in files:
        f = os.path.join(subdir, file)
        if not f.endswith('.'+extension):
          continue
        if extension == 'txt':
          dict = dict_from_txt(f)
        else:
          dict = dict_creator(f)
        dicts[curr_index] = dict
        curr_index += 1
  else:
    for index, filename in enumerate(os.listdir(directory)):
      f = os.path.join(directory, filename)
      if not f.endswith('.'+extension):
        continue
      if extension == 'txt':
        dict = dict_from_txt(f)
      else:
        dict = dict_creator(f)
      dicts[index] = dict
    
  print(dicts)
  # create a dataframe from the dictionary
  all_pars = pd.DataFrame.from_dict(dicts, orient='index')
  
  # export to csv
  all_pars.to_csv(output, index=False, sep=delimeter)
  
if __name__ == "__main__":
  parser=argparse.ArgumentParser(description="read the properties of many par files")
  parser.add_argument("--dir", type=str, default=".")
  parser.add_argument("--out", type=str, default="pars.csv")
  parser.add_argument("--sep", type=str, default=',')
  parser.add_argument("--ext", type=str, default='par')
  parser.add_argument("-r", type=bool, default=False)
  args = parser.parse_args()
  
  directory = args.dir
  output = args.out
  delimeter = args.sep
  recursive = args.r
  extension = args.ext
  
  main()    