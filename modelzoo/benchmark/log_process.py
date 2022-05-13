import time
import re
import argparse
import os
import yaml

curPath = os.path.dirname(os.path.realpath(__file__))
yamlPath = os.path.join(curPath, "config.yaml")
with open(yamlPath, 'r', encoding='utf-8') as f:
    temp = yaml.safe_load(f.read())
    WDL_batchsize=temp['model_batchsize']['wdl']
    DLRM_batchsize=temp['model_batchsize']['dlrm']
    DeepFM_batchsize=temp['model_batchsize']['deepfm']
    DSSM_batchsize=temp['model_batchsize']['dssm']
    DIEN_batchsize=temp['model_batchsize']['dien']
    DIN_batchsize=temp['model_batchsize']['din']
    print("!batchsize:\n\t WDL\tDLRM\tDeepFM\tDSSM\tDIEN\tDIN\n",'\t',WDL_batchsize,'\t',DLRM_batchsize,'\t',DeepFM_batchsize,'\t',DSSM_batchsize,'\t',DIEN_batchsize,'\t',DIN_batchsize)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        help='Full path of log directory',
                        required=False,
                        default='./')
    return parser


if __name__ == "__main__":
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
    for file in log_list:
        output = []
        file_name = os.path.split(file)[1]
        model_name = file_name.split('_')[0]
        batchsize = temp['model_batchsize'][model_name]
        file_name_nosurf = os.path.splitext(file_name)[0]
        with open(file, 'r') as f:
            for line in f:
                matchObj = re.search(r'global_step/sec: \d+(\.\d+)?', line)
                if matchObj:
                    output.append(matchObj.group()[17:])
                if "ACC" in line:
                    value = float(line.split()[2])
                    acc_dic[file_name_nosurf] = value
                    print('ACC\t   -- ',file_name_nosurf,'\t   -- ',value)
                elif "AUC" in line:
                    value = float(line.split()[2])
                    auc_dic[file_name_nosurf] = value
                    print('AUC\t   --',file_name_nosurf,'\t   -- ',value)
        
        gstep = [float(i) for i in output[20:30]]
        avg = sum(gstep) / len(gstep)
        print('Throughput --',file_name,'\t   --',avg*batchsize)
        
   
