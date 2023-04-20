import os
import json
import pandas as pd 
import re
from collections import defaultdict

glue_metrics_dict = {"sst-2":"acc", 
                "mrpc":"f1", 
                "qqp":"f1", 
                "sts-b":"corr", 
                "cola":"mcc", 
                "rte":"acc",
                "mnli":"acc",
                "qnli":"acc"}
target_folders = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-6"]

def collect_glue_metrics(logs_abs_dir, mode):
  '''
  mode in ['seed', 'regular']
  '''
  assert mode in ['seed', 'regular']
  os.chdir(logs_abs_dir)

  methods = os.listdir()
  runs = None

  tables = []

  maxes, avgs, counts = {}, {}, {}
  combined = {"File":[]}
  for metric in glue_metrics_dict.keys():
    combined[f"{metric}_avg"] = []
    combined[f"{metric}_max"] = []

  for method in methods:
    if os.path.isdir(os.path.join(logs_abs_dir,method)) and method in target_folders:
      os.chdir(os.path.join(logs_abs_dir, method))
      runs = os.listdir()
      
      for run in runs:
        if "seed" in run:
          ckpt_name = "_".join(run.split("_")[:-2])
          maxes[f"{method}/{ckpt_name}"] = defaultdict(float)
          avgs[f"{method}/{ckpt_name}"] = defaultdict(float)
          counts[f"{method}/{ckpt_name}"] = defaultdict(int)
            
      for run in runs:
        if "seed" in run and mode == 'seed':
          ckpt_name = "_".join(run.split("_")[:-2])
          seed = run[-1]
          for metric in glue_metrics_dict.keys():
            table = {"File":f"{method}/{ckpt_name}","Metric":metric,"Seed":seed}
            log_file = os.path.join(os.getcwd(), os.path.join(run, f"{metric}.log"))
            with open(log_file, 'r') as log:
              for line in log:
                if "__main__" in line and f"{glue_metrics_dict[metric]} = " in line:
                  cur_metric = line.split("-")[3].replace(" ","").replace("\n","")
                  if re.search(f"^{glue_metrics_dict[metric]}", cur_metric):
                    result = float(cur_metric.split("=")[1].replace("None",""))
                    table["Result"]=result
            tables.append(pd.DataFrame(table, index=[len(table)])) 
            maxes[f"{method}/{ckpt_name}"][metric] = max(maxes[f"{method}/{ckpt_name}"][metric], result)
            avgs[f"{method}/{ckpt_name}"][metric] += result
            counts[f"{method}/{ckpt_name}"][metric] += 1
        elif mode == 'regular':
          ckpt_name = run
          table = {"File":f"{method}/{ckpt_name}"}
          for metric in glue_metrics_dict.keys():
            # table = {"File":f"{method}/{ckpt_name}",f"{metric}:0.0}
            log_file = os.path.join(os.getcwd(), os.path.join(run, f"{metric}.log"))
            if os.path.isfile(log_file):
              with open(log_file, 'r') as log:
                for line in log:
                  if "__main__" in line and f"{glue_metrics_dict[metric]} = " in line:
                    cur_metric = line.split("-")[3].replace(" ","").replace("\n","")
                    if re.search(f"^{glue_metrics_dict[metric]}", cur_metric):
                      result= float(line.split("=")[-1].replace(" ","").replace("\n",""))
                      table[metric]=result
                      break
          tables.append(pd.DataFrame(table, index=[len(table)])) 

  if mode == 'seed':
    ret = pd.concat(tables,axis=0, ignore_index=True)
    ret.sort_values(by=["File","Seed","Metric"], inplace=True, ignore_index=True)
    ret.to_csv(os.path.join(logs_abs_dir, "metrics.csv"),index=False)
    for key, val in avgs.items():
      maxes[key], avgs[key], counts[key] = dict(maxes[key]), dict(avgs[key]), dict(counts[key])
      combined["File"].append(key)
      for k in val.keys():
        combined[f"{k}_avg"].append(avgs[key][k] / counts[key][k])
        combined[f"{k}_max"].append(maxes[key][k])
    combined_df = pd.DataFrame.from_dict(combined)
    combined_df.to_csv(os.path.join(logs_abs_dir, "metrics_max_avg.csv"), index=False)
    print(f"Results saved to {os.path.join(logs_abs_dir, 'metrics_max_avg.csv')}")

  else:
    ret = pd.concat(tables,axis=0, ignore_index=True)
    ret.sort_values(by=["File"], inplace=True, ignore_index=True, ascending=False)
    ret.to_csv(os.path.join(logs_abs_dir, "glue_metric.csv"), index=False)
    print(f"Glue results saved to {os.path.join(logs_abs_dir, 'glue_metric.csv')}")
  

def collect_squad_metrics(logs_abs_dir):
  os.chdir(logs_abs_dir)

  tables = {}

  for ckpt_folder in os.listdir():
    if ckpt_folder in target_folders:
      os.chdir(os.path.join(logs_abs_dir, ckpt_folder))
      upper_dir = os.getcwd()
      for log_folder in os.listdir():
        os.chdir(os.path.join(upper_dir, log_folder))
        logs = os.listdir()
        if "squad_log.txt" in logs:
          with open(os.path.join(os.getcwd(), "squad_log.txt"), 'r') as f:
            lines = f.readlines()
            if len(lines) > 0 and "exact_match" in lines[-1] and "F1" in lines[-1]:
              results=[result for result in lines[-1].replace("\n","").split("-")[-1].split(" ") if len(result) > 0]
              df = pd.DataFrame(data={"File_name":f"{ckpt_folder}/{log_folder}", 
                                      results[0]:results[2], 
                                      results[3]:results[5]}, index=[len(tables)])
              tables[f"{ckpt_folder}/{log_folder}"] = df
  combined = pd.concat(tables, axis=0, ignore_index=True)
  combined.sort_values(by=["File_name"], inplace=True, ignore_index=True, ascending=False)
  combined.to_csv(os.path.join(logs_abs_dir, "squad_metrics.csv"), index=False)
  print(f"SQuAD results saved to {os.path.join(logs_abs_dir, 'squad_metrics.csv')}")




if __name__ == '__main__':
  glue_logs_abs_dir = "/work/09308/zhengmk/BERT/fine-tuning/pytorch-fine-tuning/BERT-PyTorch/results/bert_pretraining/GLUE/"
  squad_logs_abs_dir = "/work/09308/zhengmk/BERT/fine-tuning/pytorch-fine-tuning/BERT-PyTorch/results/bert_pretraining/SQuAD/"
  collect_glue_metrics(glue_logs_abs_dir, 'regular')
  collect_squad_metrics(squad_logs_abs_dir)