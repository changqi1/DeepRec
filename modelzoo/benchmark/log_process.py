import time
import re
import argparse
import os
import yaml


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        help='Full path of log directory',
                        required=False,
                        default='./')
    return parser


def read_config():
    bs_dic = {}
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_path, "config.yaml")
    models=[]
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
        models  = config["test_model"]
        for model in models:
            bs_dic[model.lower()]=config['model_batchsize'][model]
        print("=====================================================================================")
        print('%10s'%'model', end="\t")
        for k in bs_dic.keys():
            print(k, end='\t')
        print("")
        print('%10s'%'batchsize' ,end="\t")
        for k in bs_dic.keys():
            print(bs_dic[k], end="\t")
        print("")
        print("=====================================================================================")
    return bs_dic, models


if __name__ == "__main__":
    bs_dic, models = read_config()
    parser = get_arg_parser()
    args = parser.parse_args()
    log_dir = args.log_dir

    log_list = []
    result={}
    for root, dirs, files in os.walk(log_dir, topdown=False):
        for name in files:
            if os.path.splitext(name)[1] == '.log':
                log_list.append(os.path.join(root, name))
    acc_dic = {}
    auc_dic = {}
    gstep_dic = {}
    for file in log_list:
        output = []
        file_name = os.path.split(file)[1]
        model_name = file_name.split('_')[0]
        file_name_nosurf = os.path.splitext(file_name)[0]
        with open(file, 'r') as f:
            for line in f:
                matchObj = re.search(r'global_step/sec: \d+(\.\d+)?', line)
                if matchObj:
                    output.append(matchObj.group()[17:])
                if "ACC" in line:
                    value = float(line.split()[2])
                    acc_dic[file_name_nosurf] = value
                if "AUC" in line:
                    value = float(line.split()[2])
                    auc_dic[file_name_nosurf] = value
    
        gstep = [float(i) for i in output[20:30]]
        avg = sum(gstep) / len(gstep)
        gstep_dic[file_name_nosurf] = avg

    print("%-30s\t %10s\t %10s\t %10s\t %10s" %('Model', 'ACC', 'AUC', 'Gstep', 'Throughput'))
    for key in gstep_dic.keys():
        model = key.split('_')[0]
        params = key.split('_')[3:]
        print("%-30s\t %10.4f\t %10.4f\t %10.4f\t %10.4f" %(key, acc_dic[key], auc_dic[key], gstep_dic[key], gstep_dic[key]*bs_dic[model]))
    

   
