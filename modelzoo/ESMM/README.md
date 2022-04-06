# ESMM

- [ESMM](#ESMM)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
    - [Distribute Training](#distribute-training)
  - [Benchmark](#benchmark)
    - [Test Environment](#test-environment)
    - [Standing-alone training](#standing-alone-training)
    - [Distribute Training](#distribute-training-1)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Fields](#fields)
    - [Processing](#processing)
  - [TODO LIST](#todo-list)

## Model Structure
[Entire Space Multi-Task Model](https://arxiv.org/abs/1804.07931)(ESMM) is a model proposed by Alibaba in 2018 that estimates post-click convertion rate(CVR) directly over the entire space. It also employes a feature representation transfer learning strategy.

The model structure in this folder is as follow:
```
output:
                                   pCTCVR
                 (post-view click-through&conversion rate)
model:
                                      |
                           _____> elem-wise x <_____
                          /                         \
                         /                           \
                       pCVR                         pCTR
            (post-view conversion rate)   (post-view click-through rate)
                   ______|______                ______|______
                   |           |                |           |
                   | Main task |                | Aux. task |
                   |    MLP    |                |    MLP    |
                   |           |                |           |
                   |___________|                |___________|
                          ^-----------      ----------^
                                      \    /
                                       \  /
                      ______________> Concat <_____________
                     /                  |                  \
                    |                   |                   |
              ______|______       ______|______       ______|______
              |           |       |           |       |           |
              |           |       |           |       |           |
              |    MLP    |       |    MLP    |       |    MLP    |
              |           |       |           |       |           |
              |___________|       |___________|       |___________|
                    |                   |                  |
              _____________       _____________       ____________
             |_Emb_|____|__|     |_Emb_|____|__|     |_Emb_|__|___|
                    |                   |                  |
input:
             [User features]     [Item features]    [Combo features]
```

## Usage

### Stand-alone Training
1.  Prepare the [data set](#prepare) first.

2.  Create a docker image by DockerFile.
    Choose DockerFile corresponding to DeepRec(Pending) or Google tensorflow.
    ```
    docker build -t DeepRec_Model_Zoo_ESMM_training:v1.0 .
    ```

3.  Run a docker container.
    ```
    docker run -it DeepRec_Model_Zoo_ESMM_training:v1.0 /bin/bash
    ```

4.  Training.
    ```
    cd /root/
    python train.py
    ```
    Use argument `--bf16` to enable DeepRec BF16 in deep model.
    ```
    python train.py --bf16
    ```
    Use arguments to set up a custom configuation:
    - `--seed`: Random seed. Default is `2021`.
    - `--data_location`: Full path of train & eval data. Default is `./data`.
    - `--steps`: Set the number of steps on train dataset. When default(`0`) is used, the number of steps is computed based on dataset size and number of epochs equaled 10.
    - `--batch_size`: Batch size to train. Default is `512`.
    - `--output_dir`: Full path to output directory for logs and saved model. Default is `./result`.
    - `--checkpoint_dir`: Full path to checkpoints output directory. Default is `$(OUTPUT_DIR)/model_$(MODEL_NAME)_$(TIMESTAMP)`
    - `--learning_rate`: Learning rate for network. Default is `0.1`.
    - `--l2_regularization`: L2 regularization for the model. Default is `None`.
    - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to `None`.
    - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to `None`.
    - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default is `1`.
    - `--bf16`: Enable DeepRec BF16 feature in DeepRec. Use FP32 by default.
    - `--no_eval`: Do not evaluate trained model by eval dataset. Evaluating model by default.
    - `--protocol`: Set the protocol("grpc", "grpc++", "star_server") used when starting server in distributed training. Default is `grpc`.
    - `--inter`: Set inter op parallelism threads. Default is `0`.
    - `--intra`: Set intra op parallelism threads. Default is `0`.
    - `--input_layer_partitioner`: Slice size of input layer partitioner(units MB). Default is `0`.
    - `--dense_layer_partitioner`: Slice size of dense layer partitioner(units kB). Default is `0`.

### Distributed Training
1. Prepare a K8S cluster and shared storage volume.
2. Create a PVC(PeritetVolumeClaim) for storage volumn in cluster.
3. Prepare docker image by DockerFile.
4. Edit k8s yaml file
    - `replicas`: numbers of chief, workers, ps;
    - `image`: place where nodes can pull the docker image from;
    - `claimName`: PVC name.


## Benchmark
### Stand-alone training

#### Test Environment
The benchmark is performed on the [Alibaba Cloud ECS general purpose instance family with high clock speeds - **ecs.hfg7.2xlarge**](https://help.aliyun.com/document_detail/25378.html?spm=5176.2020520101.vmBInfo.instanceType.4a944df5PvCcED#hfg7).
- Hardware
  - Model name:          Intel(R) Xeon(R) Platinum 8369HC CPU @ 3.30GHz
  - CPU(s):              8
  - Socket(s):           1
  - Core(s) per socket:  4
  - Thread(s) per core:  2
  - Memory:              32G

- Software
  - kernel:                 4.18.0-305.12.1.el8_4.x86_64
  - OS:                     CentOS Linux release 8.4.2105
  - Docker:                 20.10.12
  - Python:                 3.6.12

#### Performance Result

<table>
    <tr>
        <td colspan="1"></td>
        <td>Framework</td>
        <td>DType</td>
        <td>Accuracy</td>
        <td>AUC</td>
        <td>Globalsteps/sec</td>
    </tr>
    <tr>
        <td rowspan="3">ESMM</td>
        <td>Community TensorFlow</td>
        <td>FP32</td>
        <td>0.9995887</td>
        <td>0.500000</td>
        <td>121.804 (baseline)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32</td>
        <td>0.9995887</td>
        <td>0.4996916</td>
        <td>182.622 (149.93%)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32+BF16</td>
        <td>0.9995887</td>
        <td>0.4898705</td>
        <td>182.477 (149.81%)</td>
    </tr>
</table>

Community TensorFlow version is v1.15.5.

## Dataset
Taobao dataset from [EasyRec](https://github.com/AlibabaPAI/EasyRec) is used.
### Prepare
Put data file **taobao_train_data & taobao_test_data** into ./data/
For details of Data download, see [EasyRec](https://github.com/AlibabaPAI/EasyRec/#GetStarted)

### Fields
The dataset contains 20 columns, details as follow:
| Name | clk      | buy      | pid       | adgroup_id | cate_id   | campaign_id | customer  | brand     | user_id   | cms_segid | cms_group_id | final_gender_code | age_level | pvalue_level | shopping_level | occupation | new_user_class_level | tag_category_list | tag_brand_list | price    |
| ---- | -------- | -------- | --------- | ---------- | --------- | ----------- | --------- | --------- | --------- | --------- | ------------ | ----------------- | --------- | ------------ | -------------- | ---------- | -------------------- | ----------------- | -------------- | -------- |
| Type | tf.int32 | tf.int32 | tf.string | tf.string  | tf.string | tf.string   | tf.string | tf.string | tf.string | tf.string | tf.string    | tf.string         | tf.string | tf.string    | tf.string      | tf.string  | tf.string            | tf.string         | tf.string      | tf.int32 |


The data in `tag_category_list` and `tag_brand_list` column are separated by `'|'`

### Processing
The `clk` and `buy` columns are used as labels.
User's feature columns is as follow:
| Column name          | Hash bucket size | Embedding dimension |
| -------------------- | ---------------- | ------------------- |
| user_id              | 100000           | 16                  |
| cms_segid            | 100              | 16                  |
| cms_group_id         | 100              | 16                  |
| age_level            | 10               | 16                  |
| pvalue_level         | 10               | 16                  |
| shopping_level       | 10               | 16                  |
| occupation           | 10               | 16                  |
| new_user_class_level | 10               | 16                  |
| tag_category_list    | 100000           | 16                  |
| tag_brand_list       | 100000           | 16                  |

Item's feature columns is as follow:
| Column name | Hash bucket size | Number of buckets | Embedding dimension |
| ----------- | ---------------- | ------------------- | ------------------- |
| pid         | 10               | N/A                 | 16                  |
| adgroup_id  | 100000           | N/A                 | 16                  |
| cate_id     | 10000            | N/A                 | 16                  |
| campaign_id | 100000           | N/A                 | 16                  |
| customer    | 100000           | N/A                 | 16                  |
| brand       | 100000           | N/A                 | 16                  |
| price       | N/A              | 50                  | 16                  |

## TODO LIST
Next To do
- Distributed training benchmark
