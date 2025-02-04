# program which reads in all the data from a par file and puts it into a pandas dataframe
import pandas as pd
import os
import argparse


# main function
def main():
  par_cols = ["name", "value", "fitting", "error"]
  
  dicts = {}
  for index, filename in enumerate(os.listdir(directory)):
    f = os.path.join(directory, filename)
    # if file ends in .par
    if not f.endswith('.par'):
      continue
    else:
      print("Reading in file: " + f)
      
    # read in the file
    df = pd.read_csv(f, sep='\s+', header=None, names=par_cols)
    df = df.T
    df.columns = df.iloc[0]
    df.insert(0,'filename',f)
    print(df)

    # use only the data
    dict = df.to_dict(orient='records')[1]
    
    # add to the dictionary list
    dicts[index] = dict
    
  # create a dataframe from the dictionary
  all_pars = pd.DataFrame.from_dict(dicts, orient='index')
  
  # export to csv
  all_pars.to_csv(output, index=False, sep=delimeter)
  
if __name__ == "__main__":
  parser=argparse.ArgumentParser(description="read the properties of many par files")
  parser.add_argument("--dir", type=str, default="/")
  parser.add_argument("--out", type=str, default="pars.csv")
  parser.add_argument("--sep", type=str, default=',')
  args = parser.parse_args()
  
  directory = args.dir
  output = args.out
  delimeter = args.sep
  
  main()    