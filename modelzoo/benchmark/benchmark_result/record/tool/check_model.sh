# modify the shell command line to comments, by given column id or regexp
function modify_text()
{
        regex1=$1
        regex2=$2
        model_name=$3
        col_id=($(cat $src_file | grep -n $regex1 | grep $regex2  |awk -F ':' '{print $1}'))
        count=${#col_id[@]}
        for i in $(seq 1 $count)
        do
                sed -i "${col_id[i-1]}c # [Error] No model named ${model_name}, related command line has been deleted..." $src_file
        done
}


catg=$1
currentTime=$2
test_list=$3

exist_list=$(ls -F /root/modelzoo | grep '/'|sed 's/\///g')
echo $exist_list
echo $test_list
src_file=/benchmark_result/record/script/$currentTime/${catg}.sh

for model in $test_list
do
        if [[ -z $(echo $exist_list | grep $model) ]];then
                echo "echo \"[ERROR] Modle $model does not exists, related command will be skipped\"" >/benchmark_result/record/runtime/error.log
                modify_text  "echo" $model $model
                modify_text  "cd" "/root/modelzoo/$model" $model
                modify_text  "train.py" "${model,,}" $model
        fi
done
