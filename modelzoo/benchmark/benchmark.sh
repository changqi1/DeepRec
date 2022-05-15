#!/bin/bash
function echoColor() {
	case $1 in
	green)
		echo -e "\033[32;49m$2\033[0m"
		;;
	red)
		echo -e "\033[31;40m$2\033[0m"
		;;
	*)
		echo "Example: echo_color red string"
		;;
	esac
}

function echoBanner()
{
    echoColor green "#############################################################################################################################"
    echoColor green "#   ####:                       #####                       #####.                      #                           #       #"     
    echoColor green "#   #  :#.                      #    #                      #   :#                      #                           #       #"     
    echoColor green "#   #   :#  ###    ###   # ##   #    #  ###     ##:         #    #  ###   #:##:    ##:  #:##:  ## #   .###.   #:##: #  :#   #" 
    echoColor green "#   #    #    :#     :#  #   #  #   :#    :#   #            #   :#    :#  #  :#   #     #  :#  #:#:#  #: :#   ##  # # :#    #"  
    echoColor green "#   #    # #   #  #   #  #   #  #####  #   #  #.            #####. #   #  #   #  #.     #   #  # # #      #   #     #:#     #"   
    echoColor green "#   #    # #####  #####  #   #  #  .#: #####  #             #   :# #####  #   #  #      #   #  # # #  :####   #     ##      #"    
    echoColor green "#   #   :# #      #      #   #  #   .# #      #.            #    # #      #   #  #.     #   #  # # #  #:  #   #     #.#.    #"  
    echoColor green "#   #  :#.     #      #  #   #  #    #     #   #            #   :#     #  #   #   #     #   #  # # #  #.  #   #     # .#    #" 
    echoColor green "#   ####:   ###:   ###:  # ##   #    :  ###:    ##:         #####.  ###:  #   #    ##:  #   #  # # #  :##:#   #     #  :#   #" 
    echoColor green "#                        #                                                                                                  #"                                                                                               
    echoColor green "#                        #                                                                                                  #"                                                                                                
    echoColor green "#                        #                                                                                                  #"                                                                                                
    echoColor green "#############################################################################################################################"
} 

function make_script()
{
    script=$script_path

    [[ ! -d $(dirname $script) ]] && mkdir -p $(dirname $script)

    bf16_para=
    paras=$modelArgs
    
    echo "model_list=\$1" >>$script
    echo "category=\$2" >>$script
    echo "cat_param=\$3" >>$script

    echo "$env_var" >> $script && echo "">> $script

    echo " " >> $script && echo "bash  /benchmark_result/record/tool/check_model.sh $catg $currentTime \"\${model_list[*]}\"" >>$script

    for line in $model_list
    do
        log_tag=$(echo $paras| sed 's/--/_/g' | sed 's/ //g')
        [[ $paras == "" ]] && log_tag=""
        model_name=$line
        echo "echo 'testing $model_name of $catg $paras.......'" >> $script
        echo "cd /root/modelzoo/$model_name/" >> $script
        if [[ ! -d  /benchmark_result/checkpoint/$currentTime/${model_name,,}_script$$log_tag ]];then
                sudo mkdir -p /benchmark_result/checkpoint/$currentTime/${model_name,,}_$script$log_tag
        fi
        newline="LD_PRELOAD=/root/modelzoo/libjemalloc.so.2.5.1 python train.py \$cat_param $paras  --checkpoint /benchmark_result/checkpoint/$currentTime/${model_name,,}_\${category}$log_tag  >/benchmark_result/log/$currentTime/${model_name,,}_\${category}$log_tag.log 2>&1"
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
    sudo docker pull $deeprec_test_image
    runSingleContainer $deeprec_test_image deeprec_bf16
    runSingleContainer $deeprec_test_image deeprec_fp32

    if [[ $stocktf==True ]];then
        sudo docker pull $tf_test_image
        runSingleContainer $tf_test_image tf_fp32
    fi
}

function runSingleContainer()
{
    image_repo=$1
    cat_name=$2
    script_name='benchmark_modelzoo.sh'

    if [[ $cat_name == "tf_fp32" ]];then
        param="--tf"
    elif [[ $cat_name == "deeprec_bf16" ]];then
        param="--bf16"
    else
        param=
    fi

    container_name=$(echo $2 | awk -F "." '{print $1}')
    host_path=$(cd benchmark_result && pwd)
    sudo docker run -itd \
                    --name $container_name\
                    -v $host_path:/benchmark_result/\
                    $image_repo /bin/bash /benchmark_result/record/script/$currentTime/$script_name "${model_list[*]}" $cat_name $param  
}

# check container status
function checkStatus()
{
    tf_32_status=$(sudo docker inspect --format '{{.State.Running}}' tf_fp32)
    deeprec_32_status=$(sudo docker inspect --format '{{.State.Running}}' deeprec_fp32)
    deeprec_16_status=$(sudo docker inspect --format '{{.State.Running}}' deeprec_bf16)
    echo "tf32 is Running      : $tf_32_status"
    echo "deeprec32 is Running : $deeprec_32_status"
    echo "deeprec16 is Running : $deeprec_16_status"

    while [[ "$tf_32_status" == true || "${deeprec_32_status}" == true || "${deeprec_16_status}" == true ]]
    do
        tf_32_status=$(sudo docker inspect --format '{{.State.Running}}' tf_fp32)
        deeprec_32_status=$(sudo docker inspect --format '{{.State.Running}}' deeprec_fp32)
        deeprec_16_status=$(sudo docker inspect --format '{{.State.Running}}' deeprec_bf16)
       
        echo ""
        echo "------------------------------------"
        printf "%12s is Running:  %s\n" tf_fp32 $tf_32_status deeprec_fp32 $deeprec_32_status deeprec_bf16 $deeprec_16_status
        echo "------------------------------------"
        echo ""

        echo "sleep for 1 min ......"
        sleep 1m

    done
}

function main()
{
    echoBanner
    make_script\
    && checkEnv\
    && runContainers\
    && checkStatus \
    && python3 ./log_process.py --log_dir=$log_dir/$currentTime
}

# time
currentTime=`date "+%Y-%m-%d-%H-%M-%S"`

# config_file
config_file="./config.yaml"

# Args
modelArgs=$(cat $config_file | shyaml get-value modelArgs)
[[ $modelArgs == None ]] && modelArgs=

# directory
log_dir=$(cd ./benchmark_result/log/ && pwd)
checkpoint_dir=$(cd ./benchmark_result/checkpoint/ && pwd)

# run.sh
script_path="./benchmark_result/record/script/$currentTime/benchmark_modelzoo.sh"

# model list
model_list=$(cat config.yaml | shyaml get-values test_model)

# stocktf
stocktf=$(cat $config_file | shyaml get-value stocktf)

# image name
deeprec_test_image=$(cat $config_file | shyaml get-value deeprec_test_image)
tf_test_image=$(cat $config_file | shyaml get-value tf_test_image)

# environment variables
env_var=$(cat $config_file | shyaml get-values env_var)

# create dir
[ ! -d $log_dir/$currentTime ] && sudo mkdir -p "$log_dir/$currentTime"
[ ! -d $checkpoint_dir/$currentTime ] && sudo mkdir -p "$checkpoint_dir/$currentTime"

main
