import argparse
import collections
import json
import math
import numbers
import os
import tensorflow as tf
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.client import timeline
import time

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print(f'Using TensorFlow version {tf.__version__}')

@tf.custom_gradient
def round_with_gradients(x):
    def grad(dy):
        return dy
    return tf.round(x), grad

USER_COLUMN = [
    'user_id', 'cms_segid', 'cms_group_id', 'age_level', 'pvalue_level',
    'shopping_level', 'occupation', 'new_user_class_level'
]
ITEM_COLUMN = [
    'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'price'
]
COMBO_COLUMN = [
    'pid', 'tag_category_list', 'tag_brand_list'
]
LABEL_COLUMNS = ['clk', 'buy']
TAG_COLUMN = ['tag_category_list', 'tag_brand_list']

HASH_INPUTS = [
    'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand',
    'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
    'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
    'tag_category_list', 'tag_brand_list'
]

IDENTITY_INPUTS = ['price']

HASH_BUCKET_SIZES = {
        'pid': 10,
        'adgroup_id': 100000,
        'cate_id': 10000,
        'campaign_id': 100000,
        'customer': 100000,
        'brand': 100000,
        'user_id': 100000,
        'cms_segid': 100,
        'cms_group_id': 100,
        'final_gender_code': 10,
        'age_level': 10,
        'pvalue_level': 10,
        'shopping_level': 10,
        'occupation': 10,
        'new_user_class_level': 10,
        'tag_category_list': 100000,
        'tag_brand_list': 100000,
        }

NUM_BUCKETS = {
    'price': 50
}
defaults = [[0]] * len(LABEL_COLUMNS) + [[' ']] * len(HASH_INPUTS) + [[0]] * len(IDENTITY_INPUTS)
ALL_FEATURE_COLUMNS = HASH_INPUTS + IDENTITY_INPUTS
headers = LABEL_COLUMNS + ALL_FEATURE_COLUMNS
class ESMM():
    def __init__(self,
                 input,
                 user_column,
                 item_column,
                 combo_column,
                 user_mlp=[256, 128, 96, 64],
                 item_mlp=[256, 128, 96, 64],
                 combo_mlp=[128, 96, 64, 32],
                 cvr_mlp=[128, 96, 64, 32, 16],
                 ctr_mlp=[128, 96, 64, 32, 16],
                 bf16=False,
                 learning_rate=0.1,
                 l2_scale=1e-6,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not input:
            raise ValueError('Dataset is not defined.')
        if not user_column or not item_column or not combo_column:
            raise ValueError('User column, item column or combo column is not defined.')
        self.__user_column = user_column
        self.__item_column = item_column
        self.__combo_column = combo_column

        self.__user_mlp = user_mlp
        self.__item_mlp = item_mlp
        self.__combo_mlp = combo_mlp
        self.__cvr_mlp = cvr_mlp
        self.__ctr_mlp = ctr_mlp

        self.__learning_rate = learning_rate
        self.__l2_regularization = self.__l2_regularizer(l2_scale) if l2_scale else None
        self.__bf16 = bf16

        self.__input_layer_partitioner = input_layer_partitioner
        self.__dense_layer_partitioner = dense_layer_partitioner

        self.feature = input[0]
        self.label = input[1]

        self.model = self.__create_model()

        with tf.name_scope('head'):
            self.__create_loss()
            self.__create_optimizer()
            self.__create_metrics()

    # used to add summary in tensorboard
    def __add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    # compute loss
    def __create_loss(self):
        bce_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.model = tf.squeeze(self.model)
        self.loss = tf.math.reduce_mean(bce_loss_func(self.label, self.model))
        tf.summary.scalar('loss', self.loss)

    # define optimizer and generate train_op
    def __create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.__learning_rate)

        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    # compute acc & auc
    def __create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self.label,
                                                    predictions=tf.round(self.model))
        self.auc, self.auc_op = tf.metrics.auc(labels=self.label,
                                               predictions=self.model,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)
        self.__add_layer_summary(self.acc, self.acc.name)

    def __create_dense_layer(self, input, num_hidden_units, activation, layer_name):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as mlp_layer_scope:
            dense_layer = tf.layers.dense(input,
                                       units=num_hidden_units,
                                       activation=activation,
                                       kernel_regularizer=self.__l2_regularization,
                                       name=mlp_layer_scope)
            self.__add_layer_summary(dense_layer, mlp_layer_scope.name)
        return dense_layer

    def __l2_regularizer(self, scale, scope=None):
        if isinstance(scale, numbers.Integral):
            raise ValueError(f'Scale cannot be an integer: {scale}')
        if isinstance(scale, numbers.Real):
            if scale < 0.:
                raise ValueError(f'Setting a scale less than 0 on a regularizer: {scale}.')
            if scale == 0.:
                return lambda _: None

        def l2(weights):
            with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
                my_scale = tf.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
                return tf.math.multiply(my_scale, tf.nn.l2_loss(weights), name=name)

        return l2

    def __make_scope(self, name, bf16):
        if(bf16):
            return tf.variable_scope(name, reuse=tf.AUTO_REUSE).keep_weights()
        else:
            return tf.variable_scope(name, reuse=tf.AUTO_REUSE)

    # create model
    def __create_model(self):
        for key in TAG_COLUMN:
            self.feature[key] = tf.strings.split(self.feature[key], '|')

        with tf.variable_scope('user_input_layer',
                               partitioner=self.__input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            user_emb = tf.feature_column.input_layer(self.feature,
                                                     self.__user_column)
        with tf.variable_scope('item_input_layer',
                               partitioner=self.__input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            item_emb = tf.feature_column.input_layer(self.feature,
                                                     self.__item_column)
        with tf.variable_scope('combo_input_layer',
                               partitioner=self.__input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            combo_emb = tf.feature_column.input_layer(self.feature,
                                                     self.__combo_column)

        with self.__make_scope('ESMM', self.__bf16):
            if self.__bf16:
                user_emb = tf.cast(user_emb, dtype=tf.bfloat16)
                item_emb = tf.cast(item_emb, dtype=tf.bfloat16)
                combo_emb = tf.cast(combo_emb, dtype=tf.bfloat16)

            with tf.variable_scope('user_mlp_layer',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate(self.__user_mlp):
                    user_emb = self.__create_dense_layer(user_emb,
                                                         num_hidden_units,
                                                         tf.nn.relu,
                                                         f'user_mlp_{layer_id}')

            with tf.variable_scope('item_mlp_layer',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate(self.__item_mlp):
                    item_emb = self.__create_dense_layer(item_emb,
                                                         num_hidden_units,
                                                         tf.nn.relu,
                                                         f'item_mlp_{layer_id}')

            with tf.variable_scope('combo_mlp_layer',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate(self.__combo_mlp):
                    combo_emb = self.__create_dense_layer(combo_emb,
                                                          num_hidden_units,
                                                          tf.nn.relu,
                                                          f'combo_mlp_{layer_id}')

            concat = tf.concat([user_emb, item_emb, combo_emb], axis=1)

            pCVR = self.__build_cvr_model(concat)
            pCTR = self.__build_ctr_model(concat)

            pCTCVR = tf.cast(tf.multiply(round_with_gradients(pCVR), round_with_gradients(pCTR)), tf.float32)
        return pCTCVR

    def __build_cvr_model(self, net):
        with tf.variable_scope('cvr_mlp',
                               partitioner=self.__dense_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            for layer_id, num_hidden_units in enumerate(self.__cvr_mlp):
                net = self.__create_dense_layer(net,
                                                num_hidden_units,
                                                tf.nn.relu,
                                                f'cvr_mlp_hiddenlayer_{layer_id}')
            net = self.__create_dense_layer(net,
                                            1,
                                            tf.math.sigmoid,
                                            'cvr_mlp_hiddenlayer_last')
        return net

    def __build_ctr_model(self, net):
        with tf.variable_scope('ctr_mlp',
                               partitioner=self.__dense_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            for layer_id, num_hidden_units in enumerate(self.__ctr_mlp):
                net = self.__create_dense_layer(net,
                                                num_hidden_units,
                                                tf.nn.relu,
                                                f'ctr_mlp_hiddenlayer_{layer_id}')
            net = self.__create_dense_layer(net,
                                            1,
                                            tf.math.sigmoid,
                                            'ctr_mlp_hiddenlayer_last')
        return net

# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs, seed):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        columns = tf.io.decode_csv(value, record_defaults=defaults)
        all_columns = collections.OrderedDict(zip(headers, columns))
        labels = [all_columns.pop(LABEL_COLUMNS[0]), all_columns.pop(LABEL_COLUMNS[1])]
        label = tf.multiply(labels[0], labels[1])
        features = all_columns
        return features, label

    return (tf.data.TextLineDataset(filename)
            .shuffle(buffer_size=10000, seed=seed)
            .repeat(num_epochs)
            .prefetch(32)
            .batch(batch_size)
            .map(parse_csv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .prefetch(1))

# generate feature columns
def build_feature_columns():
    user_column = []
    item_column = []
    combo_column = []
    for column_name in ALL_FEATURE_COLUMNS:
        if column_name in HASH_INPUTS:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                dtype=tf.string)

            embedding_column = tf.feature_column.embedding_column(categorical_column,
                                                   dimension=16,
                                                   combiner='mean')
        elif column_name in IDENTITY_INPUTS:
            column = tf.feature_column.categorical_column_with_identity(column_name, NUM_BUCKETS[column_name])
            embedding_column = tf.feature_column.embedding_column(column,
                                                   dimension=16,
                                                   combiner='mean')
        else:
            raise ValueError('Unexpected column name occured')

        if column_name in USER_COLUMN:
            user_column.append(embedding_column)
        elif column_name in ITEM_COLUMN:
            item_column.append(embedding_column)
        elif column_name in COMBO_COLUMN:
            combo_column.append(embedding_column)

    return user_column, item_column, combo_column

def train(sess_config,
          input_hooks,
          model,
          train_init_op,
          train_steps,
          keep_checkpoint_max,
          checkpoint_dir,
          save_steps=None,
          timeline_steps=None,
          no_eval=None,
          tf_config=None,
          server=None):
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), train_init_op),
        saver=tf.train.Saver(max_to_keep=keep_checkpoint_max))

    stop_hook = tf.train.StopAtStepHook(last_step=train_steps)
    log_hook = tf.train.LoggingTensorHook(
        {'steps': model.global_step,
         'loss': model.loss}, every_n_iter=100)
    hooks.append(stop_hook)
    hooks.append(log_hook)

    if timeline_steps and timeline_steps > 0:
        hooks.append(tf.train.ProfilerHook(save_steps=timeline_steps,
                                           output_dir=checkpoint_dir))
    save_ckp_steps = save_steps if save_steps or no_eval else train_steps

    with tf.train.MonitoredTrainingSession(
            master=server.target if server else '',
            is_chief=tf_config['is_chief'] if tf_config else True,
            hooks=hooks,
            scaffold=scaffold,
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=save_ckp_steps,
            summary_dir=checkpoint_dir,
            save_summaries_steps=save_steps,
            config=sess_config) as sess:
        while not sess.should_stop():
            sess.run([model.loss, model.train_op])
    print("Training completed.")


def eval(sess_config, input_hooks, model, test_init_op, test_steps, output_dir, checkpoint_dir):
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), test_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(output_dir, 'eval'))
    merged = tf.summary.merge_all()

    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        for _in in range(1, test_steps + 1):
            if (_in != test_steps):
                sess.run([model.acc_op, model.auc_op])
                if (_in % 1000 == 0):
                    print(f'Evaluation complete:[{_in}/{test_steps}]')
            else:
                eval_acc, eval_auc, events = sess.run([model.acc_op, model.auc_op, merged])
                writer.add_summary(events, _in)
                print(f'Evaluation complete:[{_in}/{test_steps}]')
                print(f'ACC = {eval_acc}\nAUC = {eval_auc}')

def main(tf_config=None, server=None):
    # check dataset and count data set size
    print("Checking dataset...")
    train_file = os.path.join(args.data_location, 'taobao_train_data')
    test_file = os.path.join(args.data_location, 'taobao_test_data')
    if not os.path.exists(args.data_location):
        raise ValueError(f'[ERROR] data location: {args.data_location} does not exist. '
                         'Please provide valid path')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise ValueError('[ERROR] taobao_train_data or taobao_test_data does not exist '
                         'in the given data_location. Please provide valid path')

    no_of_training_examples = sum(1 for _ in open(train_file))
    no_of_test_examples = sum(1 for _ in open(test_file))

    # set batch size, eporch & steps
    batch_size = args.batch_size
    if args.steps == 0:
        no_epochs = 10
        train_steps = math.ceil(
            (float(no_epochs) * no_of_training_examples) / batch_size)
    else:
        no_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)

    print(f'Numbers of training dataset: {no_of_training_examples}')
    print(f'Number of epochs: {no_epochs}')
    print(f'Number of train steps: {train_steps}')

    print(f'Numbers of test dataset: {no_of_test_examples}')
    print(f'Numbers of test steps: {test_steps}')

    # set fixed random seed
    SEED = args.seed
    tf.set_random_seed(SEED)

    # set directory path for checkpoint_dir
    output_dir = os.path.join(args.output_dir, 'model_ESMM_' + str(int(time.time())))
    print(f'Saving model events to {args.output_dir}')

    keep_checkpoint_max = args.keep_checkpoint_max
    checkpoint_dir = args.checkpoint_dir
    if args.checkpoint_dir:
        print(f'Saving checkpoint to {args.checkpoint_dir}. '
              f'Maximum number of saved checkpoints: {keep_checkpoint_max}')
    elif not args.checkpoint_dir and args.save_steps:
        print(f'Saving checkpoint to {args.output_dir}. '
              f'Maximum number of saved checkpoints: {keep_checkpoint_max}')
        checkpoint_dir = args.output_dir

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_epochs, SEED)
    test_dataset = build_model_input(test_file, batch_size, 1, SEED)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create feature column
    user_column, item_column, combo_column = build_feature_columns()

    # create variable partitioner for distributed training
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
                                max_partitions=num_ps_replicas,
                                min_slice_size=args.input_layer_partitioner <<
                                    20) if args.input_layer_partitioner else None
    dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
                                max_partitions=num_ps_replicas,
                                min_slice_size=args.dense_layer_partitioner <<
                                    10) if args.dense_layer_partitioner else None


    # create model
    model = ESMM(next_element,
                 user_column,
                 item_column,
                 combo_column,
                 bf16=args.bf16,
                 learning_rate=args.learning_rate,
                 l2_scale=args.l2_regularization,
                 input_layer_partitioner=input_layer_partitioner,
                 dense_layer_partitioner=dense_layer_partitioner)

    # Session config
    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    # Session hook
    hooks = []

    # Run model training and evaluation
    train(sess_config, hooks, model, train_init_op, train_steps,
          keep_checkpoint_max, checkpoint_dir, args.save_steps, args.timeline, args.no_eval, tf_config, server)
    if not (args.no_eval or tf_config):
        eval(sess_config, hooks, model, test_init_op, test_steps,
             output_dir, checkpoint_dir)

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='set random seed', type=int, default=2021)
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        default='./result')
    parser.add_argument('--checkpoint_dir',
                        help='Full path to checkpoints output directory')
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.1)
    parser.add_argument('--l2_regularization',
                        help='L2 regularization for the model',
                        type=float)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline',
                        type=int)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset',
                        action='store_true')
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc'],
                        default='grpc')
    parser.add_argument('--inter',
                        help='set inter op parallelism threads',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set intra op parallelism threads',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner',
                        help='slice size of input layer partitioner. units MB',
                        type=int,
                        default=0)
    parser.add_argument('--dense_layer_partitioner',
                        help='slice size of dense layer partitioner. units KB',
                        type=int,
                        default=0)
    return parser

# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    print(f'Running distributed training with TF_CONFIG: {TF_CONFIG}')

    tf_config = json.loads(TF_CONFIG)
    cluster_config = tf_config.get('cluster')
    ps_hosts = []
    worker_hosts = []
    chief_hosts = []
    for key, value in cluster_config.items():
        if 'ps' == key:
            ps_hosts = value
        elif 'worker' == key:
            worker_hosts = value
        elif 'chief' == key:
            chief_hosts = value
    if chief_hosts:
        worker_hosts = chief_hosts + worker_hosts

    if not ps_hosts or not worker_hosts:
        raise ValueError(f'[TF_CONFIG ERROR] Incorrect ps_hosts or incorrect worker_hosts')

    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol=args.protocol)
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {'ps_hosts': ps_hosts,
                     'worker_hosts': worker_hosts,
                     'type': task_type,
                     'index': task_index,
                     'is_chief': is_chief}

        tf_device = tf.device(tf.train.replica_device_setter(
                              worker_device=f'/job:worker/task:{task_index}',
                              cluster=cluster))
        return tf_config, server, tf_device
    else:
        raise ValueError(f'[TF_CONFIG ERROR] Task type or index error.')

# Some DeepRec's features are enabled by ENV.
# This func is used to set ENV and enable these features.
# A triple quotes comment is used to introduce these features and play an emphasizing role.
def set_env_for_DeepRec():
    '''
    Set some ENV for these DeepRec's features enabled by ENV.
    More Detail information is shown in https://deeprec.readthedocs.io/zh/latest/index.html.
    START_STATISTIC_STEP & STOP_STATISTIC_STEP: On CPU platform, DeepRec supports memory optimization
        in both stand-alone and distributed trainging. It's default to open, and the
        default start and stop steps of collection is 1000 and 1100. Reduce the initial
        cold start time by the following settings.
    MALLOC_CONF: On CPU platform, DeepRec can use memory optimization with the jemalloc library.
        Please preload libjemalloc.so by `LD_PRELOAD=./libjemalloc.so.2 python ...`
    DNNL_MAX_CPU_ISA: Specify the highest instruction set used by oneDNN (when the version is less than 2.5.0),
        it will be set to AVX512_CORE_AMX to enable Intel CPU's feature.
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF'] = \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    #if not args.tf:
    #    set_env_for_DeepRec()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        print('Running stand-alone mode training')
        main()
    else:
        tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
        main(tf_config, server)
