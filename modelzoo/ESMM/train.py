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

INPUT_COLUMN = [
    'clk', 'buy', 'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer',
    'brand', 'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
    'age_level', 'pvalue_level', 'shopping_level', 'occupation',
    'new_user_class_level', 'tag_category_list', 'tag_brand_list', 'price'
]
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
LABEL_COLUMN = ['clk', 'buy']
TAG_COLUMN = ['tag_category_list', 'tag_brand_list']
INPUT_FEATURES = {
    'pid': {
        'type': 'IdFeature',
        'hash_bucket_size': 10
    },
    'adgroup_id': {
        'type': 'IdFeature',
        'hash_bucket_size': 100000
    },
    'cate_id': {
        'type': 'IdFeature',
        'hash_bucket_size': 10000
    },
    'campaign_id': {
        'type': 'IdFeature',
        'hash_bucket_size': 100000
    },
    'customer': {
        'type': 'IdFeature',
        'hash_bucket_size': 100000
    },
    'brand': {
        'type': 'IdFeature',
        'hash_bucket_size': 100000
    },
    'user_id': {
        'type': 'IdFeature',
        'hash_bucket_size': 100000
    },
    'cms_segid': {
        'type': 'IdFeature',
        'hash_bucket_size': 100
    },
    'cms_group_id': {
        'type': 'IdFeature',
        'hash_bucket_size': 100
    },
    'final_gender_code': {
        'type': 'IdFeature',
        'hash_bucket_size': 10
    },
    'age_level': {
        'type': 'IdFeature',
        'hash_bucket_size': 10
    },
    'pvalue_level': {
        'type': 'IdFeature',
        'hash_bucket_size': 10
    },
    'shopping_level': {
        'type': 'IdFeature',
        'hash_bucket_size': 10
    },
    'occupation': {
        'type': 'IdFeature',
        'hash_bucket_size': 10
    },
    'new_user_class_level': {
        'type': 'IdFeature',
        'hash_bucket_size': 10
    },
    'tag_category_list': {
        'type': 'TagFeature',
        'hash_bucket_size': 100000
    },
    'tag_brand_list': {
        'type': 'TagFeature',
        'hash_bucket_size': 100000
    },
    'price': {
        'type': 'IdFeature',
        'hash_bucket_size': 50
        # 'num_buckets': 50
        # num_buckets is more consistent with easy rec's
        # implementation but outputs lower AUC for earlier steps
    },
}

def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)

def generate_input_data(filename, batch_size, num_epochs, seed):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        string_defaults = [[' '] for i in range(1, 19)]
        label_defaults = [[0], [0]]
        column_headers = INPUT_COLUMN
        record_defaults = label_defaults + string_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = [all_columns.pop(LABEL_COLUMN[0]), all_columns.pop(LABEL_COLUMN[1])]
        label = tf.multiply(labels[0], labels[1])
        # Line below is required if 'price' is using num_buckets
        # all_columns['price'] = tf.strings.to_number(all_columns['price'], out_type=tf.int32)
        features = all_columns
        return features, label

    return (tf.data.TextLineDataset(filename)
            .shuffle(buffer_size=10000, seed=seed)
            .repeat(num_epochs)
            .prefetch(32)
            .batch(batch_size)
            .map(parse_csv, num_parallel_calls=28)
            .prefetch(1))

def build_feature_cols():
    user_column = []
    item_column = []
    combo_column = []
    for key in INPUT_FEATURES:
        # Lines below is required if 'price' is using num_buckets
        # if key == 'price':
        #    categorical_column = tf.feature_column.categorical_column_with_identity(
        #        key,
        #        num_buckets=INPUT_FEATURES[key]['num_buckets'])
        # else:
        categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key,
                hash_bucket_size=INPUT_FEATURES[key]['hash_bucket_size'],
                dtype=tf.string)
        embedding_column = tf.feature_column.embedding_column(categorical_column,
                                                              dimension=16,
                                                              combiner='mean')
        if key in USER_COLUMN:
            user_column.append(embedding_column)
        elif key in ITEM_COLUMN:
            item_column.append(embedding_column)
        elif key in COMBO_COLUMN:
            combo_column.append(embedding_column)

    return user_column, item_column, combo_column

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

        self.model = self.__build_model()

        with tf.name_scope('head'):
            self.train_op, self.loss = self.compute_loss()
            self.acc, self.acc_op = tf.metrics.accuracy(labels=self.label,
                                                        predictions=self.model)
            self.auc, self.auc_op = tf.metrics.auc(labels=self.label,
                                                   predictions=self.model,
                                                   num_thresholds=1000)
            tf.summary.scalar('eval_acc', self.acc)
            tf.summary.scalar('eval_auc', self.auc)

    def compute_loss(self):
        bce_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.model = tf.squeeze(self.model)
        loss = tf.math.reduce_mean(bce_loss_func(self.label, self.model))
        tf.summary.scalar('loss', loss)

        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.__learning_rate)

        train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op, loss

    def __create_dense_layer(self, input, num_hidden_units, activation, layer_name):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as mlp_layer_scope:
            dense_layer = tf.layers.dense(input,
                                       units=num_hidden_units,
                                       activation=activation,
                                       kernel_regularizer=self.__l2_regularization,
                                       name=mlp_layer_scope)
            add_layer_summary(dense_layer, mlp_layer_scope.name)
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

    def __build_model(self):
        #for key in TAG_COLUMN:
        #    self.feature[key] = tf.strings.split(self.feature[key], '|')

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

        if self.__bf16:
            user_emb = tf.cast(user_emb, dtype=tf.bfloat16)
            with tf.variable_scope('user_mlp_layer',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE).keep_weights():
                for layer_id, num_hidden_units in enumerate(self.__user_mlp):
                    user_emb = self.__create_dense_layer(user_emb,
                                                          num_hidden_units,
                                                          tf.nn.relu,
                                                          f'user_mlp_{layer_id}')
            user_emb = tf.cast(user_emb, dtype=tf.float32)
        else:
            with tf.variable_scope('user_mlp_layer',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate(self.__user_mlp):
                    user_emb = self.__create_dense_layer(user_emb,
                                                         num_hidden_units,
                                                         tf.nn.relu,
                                                         f'user_mlp_{layer_id}')

        if self.__bf16:
            item_emb = tf.cast(item_emb, dtype=tf.bfloat16)
            with tf.variable_scope('item_mlp_layer',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE).keep_weights():
                for layer_id, num_hidden_units in enumerate(self.__item_mlp):
                    item_emb = self.__create_dense_layer(item_emb,
                                                         num_hidden_units,
                                                         tf.nn.relu,
                                                         f'item_mlp_{layer_id}')
            item_emb = tf.cast(item_emb, dtype=tf.float32)
        else:
            with tf.variable_scope('item_mlp_layer',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate(self.__item_mlp):
                    item_emb = self.__create_dense_layer(item_emb,
                                                         num_hidden_units,
                                                         tf.nn.relu,
                                                         f'item_mlp_{layer_id}')

        if self.__bf16:
            combo_emb = tf.cast(combo_emb, dtype=tf.bfloat16)
            with tf.variable_scope('combo_mlp_layer',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE).keep_weights():
                for layer_id, num_hidden_units in enumerate(self.__combo_mlp):
                    combo_emb = self.__create_dense_layer(combo_emb,
                                                          num_hidden_units,
                                                          tf.nn.relu,
                                                          f'combo_mlp_{layer_id}')
            combo_emb = tf.cast(combo_emb, dtype=tf.float32)
        else:
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

        pCTCVR = tf.cast(tf.multiply(pCVR, pCTR), tf.float32)
        return pCTCVR

    def __build_cvr_model(self, net):
        if self.__bf16:
            net = tf.cast(net, dtype=tf.bfloat16)
            with tf.variable_scope('cvr_mlp',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE).keep_weights():
                for layer_id, num_hidden_units in enumerate(self.__cvr_mlp):
                    net = self.__create_dense_layer(net,
                                                    num_hidden_units,
                                                    tf.nn.relu,
                                                    f'cvr_mlp_hiddenlayer_{layer_id}')
                net = self.__create_dense_layer(net,
                                                1,
                                                tf.math.sigmoid,
                                                'cvr_mlp_hiddenlayer_last')
            net = tf.cast(net, dtype=tf.float32)
        else:
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
        if self.__bf16:
            net = tf.cast(net, dtype=tf.bfloat16)
            with tf.variable_scope('ctr_mlp',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE).keep_weights():
                for layer_id, num_hidden_units in enumerate(self.__ctr_mlp):
                    net = self.__create_dense_layer(net,
                                                    num_hidden_units,
                                                    tf.nn.relu,
                                                    f'ctr_mlp_hiddenlayer_{layer_id}')
                net = self.__create_dense_layer(net,
                                                1,
                                                tf.math.sigmoid,
                                                'ctr_mlp_hiddenlayer_last')
            net = tf.cast(net, dtype=tf.float32)
        else:
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

def train(model,
          output_dir,
          train_dataset,
          test_dataset,
          keep_checkpoint_max,
          train_steps,
          test_steps,
          no_eval=False,
          timeline_steps=None,
          save_steps=None,
          checkpoint_dir=None,
          tf_config=None,
          server=None,
          inter=None,
          intra=None
          ):
    output_dir = os.path.join(output_dir, 'model_ESMM_' + str(int(time.time())))
    print(f'Saving model events to {output_dir}')

    if checkpoint_dir:
        print(f'Saving checkpoint to {checkpoint_dir}. '
              f'Maximum number of saved checkpoints: {keep_checkpoint_max}')
    elif not checkpoint_dir and save_steps:
        print(f'Saving checkpoint to {output_dir}. '
              f'Maximum number of saved checkpoints: {keep_checkpoint_max}')
        checkpoint_dir = output_dir

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    sess_config = tf.ConfigProto()

    if tf_config:
        if inter:
            sess_config.inter_op_parallelism_threads = inter
        if intra:
            sess_config.intra_op_parallelism_threads = intra
        hooks = []

        hooks.append(tf.train.StopAtStepHook(last_step=train_steps))
        hooks.append(tf.train.LoggingTensorHook(
                        {
                            'steps': model.global_step,
                            'loss': model.loss
                        },
                     every_n_iter=100))

        scaffold = tf.train.Scaffold(
            local_init_op=tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer(), train_init_op))

        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=tf_config['is_chief'],
                checkpoint_dir=checkpoint_dir,
                scaffold=scaffold,
                hooks=hooks,
                log_step_count_steps=100,
                config=sess_config) as sess:
            while not sess.should_stop():
                _, train_loss = sess.run([model.train_op, model.loss])
    else:
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(output_dir, sess.graph)
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=keep_checkpoint_max)

            # train model
            sess.run(train_init_op)

            start = time.perf_counter()
            for _in in range(0, train_steps):
                if ((save_steps and save_steps > 0 and (_in % save_steps == 0 or _in == train_steps - 1))
                    or (checkpoint_dir and not save_steps and _in == train_steps - 1)):
                    _, train_loss, events = sess.run([model.train_op, model.loss, merged])
                    writer.add_summary(events, _in)
                    checkpoint_path = saver.save(sess,
                                                 save_path=os.path.join(
                                                 checkpoint_dir,
                                                 'esmm-checkpoint'),
                                                 global_step=_in)
                    print(f'Saved checkpoint to {checkpoint_path}')
                elif timeline_steps and timeline_steps > 0 and _in % timeline_steps == 0:
                    _, train_loss = sess.run([model.train_op, model.loss],
                                             options=options,
                                             run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    print(f'Saved timeline to {output_dir}')
                    with open(os.path.join(output_dir, f'timeline-{_in}.json'), 'w') as f:
                        f.write(chrome_trace)
                else:
                    _, train_loss = sess.run([model.train_op, model.loss])

                # print training loss and time cost
                if (_in % 100 == 0 or _in == train_steps - 1):
                    end = time.perf_counter()
                    cost_time = end - start
                    global_step_sec = (100 if _in % 100 == 0 else train_steps -
                                       1 % 100) / cost_time
                    print(f'global_step/sec: {global_step_sec:.4f}')
                    print(f'loss = {train_loss}, steps = {_in}, cost time = {cost_time:0.2f}s')
                    start = time.perf_counter()

            # eval model
            if not no_eval:
                writer = tf.summary.FileWriter(os.path.join(output_dir, 'eval'))

                sess.run(test_init_op)
                for _in in range(1, test_steps + 1):
                    if (_in != test_steps):
                        sess.run([model.acc, model.acc_op, model.auc, model.auc_op])
                        if (_in % 1000 == 0):
                            print(f'Evaluation complete:[{_in}/{test_steps}]')
                    else:
                        eval_acc, _, eval_auc, _, events = sess.run([model.acc,
                                                                     model.acc_op,
                                                                     model.auc,
                                                                     model.auc_op,
                                                                     merged])
                        writer.add_summary(events, _in)
                        print(f'Evaluation complete:[{_in}/{test_steps}]')
                        print(f'ACC = {eval_acc}\nAUC = {eval_auc}')

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

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    SEED = args.seed
    tf.set_random_seed(SEED)

    print('Validating arguments')
    train_file = os.path.join(args.data_location, 'taobao_train_data')
    test_file = os.path.join(args.data_location, 'taobao_test_data')
    if not os.path.exists(args.data_location):
        raise ValueError(f'[ERROR] data location: {args.data_location} does not exist. '
                         'Please provide valid path')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise ValueError('[ERROR] taobao_train_data or taobao_test_data does not exist '
                         'in the given data_location. Please provide valid path')

    if not os.path.exists(args.output_dir):
        raise ValueError(f'[ERROR] output dir: {args.output_dir} does not exist. '
                         'Please provide valid path')

    no_of_training_examples = sum(1 for _ in open(train_file))
    no_of_test_examples = sum(1 for _ in open(test_file))

    # set params
    # set batch size & steps
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

    TF_CONFIG = os.getenv('TF_CONFIG')
    if TF_CONFIG:
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
        if is_chief:
            print('This host is a chief')
        cluster = tf.train.ClusterSpec({
            'ps': ps_hosts,
            'worker': worker_hosts
        })
        server = tf.distribute.Server(cluster,
                                      job_name=task_type,
                                      task_index=task_index,
                                      protocol=args.protocol)

        if task_type == 'ps':
            server.join()
        elif task_type == 'worker':
            with tf.device(tf.train.replica_device_setter(
                                worker_device=f'/job:worker/task:{task_index}',
                                cluster=cluster)):
                tf_config={'ps_hosts': ps_hosts,
                           'worker_hosts': worker_hosts,
                           'type': task_type,
                           'index': task_index,
                           'is_chief': is_chief}

                user_column, item_column, combo_column = build_feature_cols()
                train_dataset = generate_input_data(train_file, batch_size, no_epochs, SEED)
                test_dataset = generate_input_data(test_file, batch_size, 1, SEED)

                iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                           train_dataset.output_shapes)
                next_element = iterator.get_next()

                # create variable partitioner for distributed training
                num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
                input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
                                            max_partitions=num_ps_replicas,
                                            min_slice_size=args.input_layer_partitioner <<
                                                20) if args.input_layer_partitioner else None
                dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
                                            max_partitions=num_ps_replicas,
                                            min_slice_size=args.dense_layer_partitioner <<
                                                10) if args.dense_layer_partitioner else None

                model = ESMM(next_element,
                             user_column,
                             item_column,
                             combo_column,
                             bf16=args.bf16,
                             learning_rate=args.learning_rate,
                             l2_scale=args.l2_regularization,
                             input_layer_partitioner=input_layer_partitioner,
                             dense_layer_partitioner=dense_layer_partitioner)

                train(model,
                      args.output_dir,
                      train_dataset,
                      test_dataset,
                      args.keep_checkpoint_max,
                      train_steps,
                      test_steps,
                      no_eval=args.no_eval,
                      timeline_steps=args.timeline,
                      save_steps=args.save_steps,
                      checkpoint_dir=args.checkpoint_dir,
                      tf_config=tf_config,
                      server=server,
                      inter=args.inter,
                      intra=args.intra)


    else:
        print('Running stand-alone mode training')

        user_column, item_column, combo_column = build_feature_cols()
        train_dataset = generate_input_data(train_file, batch_size, no_epochs, SEED)
        test_dataset = generate_input_data(test_file, batch_size, 1, SEED)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)
        next_element = iterator.get_next()

        model = ESMM(next_element,
                     user_column,
                     item_column,
                     combo_column,
                     bf16=args.bf16,
                     learning_rate=args.learning_rate,
                     l2_scale=args.l2_regularization)

        train(model,
              args.output_dir,
              train_dataset,
              test_dataset,
              args.keep_checkpoint_max,
              train_steps,
              test_steps,
              no_eval=args.no_eval,
              timeline_steps=args.timeline,
              save_steps=args.save_steps,
              checkpoint_dir=args.checkpoint_dir)
