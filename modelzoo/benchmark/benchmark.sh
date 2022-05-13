# script
function make_script()
{

	IFS_old=$IFS
	IFS=$'\n'	
	make_single_script deeprec_bf16 $deeprec_bf16_script
	make_single_script deeprec_fp32 $deeprec_fp32_script
	make_single_script tf_fp32 $tf_fp32_script
	IFS=$IFS_old
}
function make_single_script()
{
    catg=$1
    script=$2
    bf16_para=
    if [[ ! -d $(dirname $script) ]];then
        mkdir -p $(dirname $script)
    fi
    if [[ $modelArgs==None ]];then
        paras=" "
    else
        paras=$modelArgs
    fi    
    echo "model_list=\$model_list" >>$script
    [[ $catg != "tf_fp32" ]] &&echo " " >> $script &&  echo "$env_var" >> $script
    echo " " >> $script && echo "bash  /benchmark_result/record/tool/check_model.sh $catg $currentTime "${model_list[*]}"" >>$script
    [[ $catg == "deeprec_bf16" ]] && bf16_para="--bf16"
    [[ $catg == "tf_fp32" ]] && tf_para="--tf"
    for line in $model_list
    do
        log_tag=$(echo $paras| sed 's/ /_/g')
        model_name=$line
        echo "echo 'testing $model_name of $catg $paras.......'" >> $script
        echo "cd /root/modelzoo/$model_name/" >> $script
        if [[ ! -d  $checkpoint_dir$currentTime/${model_name,,}_script$$log_tag ]];then
                sudo mkdir -p $checkpoint_dir$currentTime/${model_name,,}_$script$log_tag
        fi
        newline="LD_PRELOAD=/root/modelzoo/libjemalloc.so.2.5.1 python train.py $paras $tf_para $bf16_para --checkpoint $checkpoint_dir$currentTime/${model_name,,}_$catg$log_tag  >$log_dir$currentTime/${model_name,,}_$catg$log_tag.log 2>&1"
        echo $newline >> $script
    done
}


# check container environment
function checkEnv()
{
    status1=$(sudo docker ps -a | grep deeprec_bf16)
    status2=$(sudo docker ps -a | grep deeprec_fp32)
    status3=$(sudo docker ps -a | grep tf_fp32)
    if [[  -n $status1 ]];then
        sudo docker rm -f deeprec_bf16
    fi
    if [[  -n $status2 ]];then
        sudo docker rm -f deeprec_fp32
    fi
    if [[  -n $status3 ]];then
        sudo docker rm -f tf_fp32
    fi
}


# run containers
function runContainers()
{
    runSingleContainer $deeprec_test_image deeprec_bf16.sh
    runSingleContainer $deeprec_test_image deeprec_fp32.sh
    if [[ $stocktf_test==True ]];then
        runSingleContainer $tf_test_image tf_fp32.sh
    fi
}
function runSingleContainer()
{
    image_repo=$1
    script_name=$2
    container_name=$(echo $2 | awk -F "." '{print $1}')
    host_path=$(cd benchmark_result && pwd)
    sudo docker run --name $container_name\
                    --rm \
                    -v $host_path:/benchmark_result/\
                    $image_repo /bin/bash /benchmark_result/record/script/$currentTime/$script_name "${model_list[*]}"
}

# check container status
function checkStatus()
{
    echo "sleep for 2 min ....."
    sleep 2m
    tf_32_status=$(sudo docker ps -a |grep tf_fp32| awk -F " " '{print $8$9}')
    deeprec_32_status=$(sudo docker ps -a |grep deeprec_fp32| awk -F " " '{print $8$9}')
    deeprec_16_status=$(sudo docker ps -a |grep deeprec_bf16| awk -F " " '{print $8$9}')
    echo "tf32:${tf_32_status}"
    echo "deeprec32:${deeprec_32_status}"
    echo "deeprec16:${deeprec_16_status}"

    while [[ "$tf_32_status" == *"Up"* || "${deeprec_32_status}" == *"Up"* || "${deeprec_16_status}" == *"Up"* ]]
    do
        tf_32_status=$(sudo docker ps -a |grep tf_fp32| awk -F " " '{print $6$7$8$9$10}')
        deeprec_32_status=$(sudo docker ps -a |grep deeprec_fp32| awk -F " " '{print $6$7$8$9$10}')
        deeprec_16_status=$(sudo docker ps -a |grep deeprec_bf16| awk -F " " '{print $6$7$8$9$10}')

        echo "--------------------------------------------------"
        echo "the status of tf_fp32 is $tf_32_status)..."
        echo "the status of deeprec_32 is $deeprec_32_status..."
        echo "the status of deeprec_16 is $deeprec_16_status..."
        echo "--------------------------------------------------"
        echo ""

        echo "sleep for 5 min ......"
        sleep 5m

    done
}

# time
currentTime=`date "+%Y-%m-%d-%H-%M-%S"`

# config_file
config_file="./config.yaml"

# modelArgs
modelArgs=$(cat $config_file | shyaml get-value modelArgs)

# log/checkpoint dir in image(log&checkpoint) and host(gol&pointcheck)
log_dir=$(cat $config_file | shyaml get-value log_dir)
checkpoint_dir=$(cat $config_file | shyaml get-value checkpoint_dir)
gol_dir=$(cat $config_file | shyaml get-value gol_dir)
pointcheck_dir=$(cat $config_file | shyaml get-value pointcheck_dir)

# run.sh
deeprec_fp32_script="./benchmark_result/record/script/$currentTime/deeprec_fp32.sh"
deeprec_bf16_script="./benchmark_result/record/script/$currentTime/deeprec_bf16.sh"
tf_fp32_script="./benchmark_result/record/script/$currentTime/tf_fp32.sh"

# model list
model_list=$(cat config.yaml | shyaml get-values test_model)

# stocktf
stocktf_test=$(cat $config_file | shyaml get-value stocktf_test)

# image name
deeprec_test_image=$(cat $config_file | shyaml get-value deeprec_test_image)
tf_test_image=$(cat $config_file | shyaml get-value tf_test_image)

# pull image
sudo docker pull $deeprec_test_image
sudo docker pull $tf_test_image

# env
env_var=$(cat $config_file | shyaml get-values env_var)

# create dir
if [ ! -d $gol_dir$currentTime ];then
  sudo mkdir -p "$gol_dir$currentTime"
fi
if [ ! -d $pointcheck_dir$currentTime ];then
  sudo mkdir -p "$pointcheck_dir$currentTime"
fi

make_script\
&& checkEnv\
&& runContainers\
&& checkStatus \
&& python3 ./log_process.py --log_dir=$gol_dir/$currentTime \
