import argparse
import collections
import json
import math
import os
import tensorflow as tf
from tensorflow.python.client import timeline as tf_timeline
from tensorflow.python.ops import partitioned_variables
import time

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print(f'Using TensorFlow version {tf.__version__}')

LABEL_COLUMNS = ['clk', 'buy']
HASH_INPUTS = [
        'pid',
        'adgroup_id',
        'cate_id',
        'campaign_id',
        'customer',
        'brand',
        'user_id',
        'cms_segid',
        'cms_group_id',
        'final_gender_code',
        'age_level',
        'pvalue_level',
        'shopping_level',
        'occupation',
        'new_user_class_level',
        'tag_category_list',
        'tag_brand_list'
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
        'tag_brand_list': 100000
        }
defaults = [[0]] * len(LABEL_COLUMNS) + [[' ']] * len(HASH_INPUTS) + [[0]] * len(IDENTITY_INPUTS)
headers = LABEL_COLUMNS + HASH_INPUTS + IDENTITY_INPUTS

def build_feature_cols():
    feature_columns = []
    for i in range(len(HASH_INPUTS)):
        feature_columns.append(
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_hash_bucket(
                        HASH_INPUTS[i],
                        hash_bucket_size=HASH_BUCKET_SIZES[HASH_INPUTS[i]],
                        dtype=tf.string
                        ),
                    dimension=16,
                    combiner='mean'))

    for i in range(len(IDENTITY_INPUTS)):
        tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(IDENTITY_INPUTS[i], 50),
                dimension=16,
                combiner='mean')

    return feature_columns

def generate_input_data(filename, batch_size, num_epochs, seed):
    def parse_csv(x):
        l = list(zip(headers, tf.io.decode_csv(x, defaults)))
        # This is because Dataset.map() have strange requirement of using collections.OrderedDict
        # otherwise throws type exception.
        return collections.OrderedDict(l[:2]), collections.OrderedDict(l[2:])

    return (tf.data.TextLineDataset(filename)
            .shuffle(buffer_size=400000, seed=seed)
            .repeat(num_epochs)
            .batch(batch_size)
            .map(parse_csv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .prefetch(32))

def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)

class SimpleMultiTask():
    def __init__(self,
                 input,
                 feature_columns,
                 mlp=[256, 196, 128, 64],
                 bf16=False,
                 learning_rate=0.1,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not input:
            raise ValueError('Dataset is not defined.')
        if not feature_columns:
            raise ValueError('Feature columns are not defined.')

        self.__feature_columns = feature_columns
        self.__mlp = mlp
        self.__bf16 = bf16
        self.__learning_rate = learning_rate

        self.__input_layer_partitioner = input_layer_partitioner
        self.__dense_layer_partitioner = dense_layer_partitioner

        self.__feature = input[1]
        self.__labels = input[0]
        self.__label = tf.stack([self.__labels['clk'], self.__labels['buy']], axis=1)

        with tf.variable_scope('input_layer',
                                partitioner=self.__input_layer_partitioner,
                                reuse=tf.AUTO_REUSE):
            self.__input_layer = tf.feature_column.input_layer(self.__feature, self.__feature_columns)
            if self.__bf16:
                self.__input_layer = tf.cast(self.__input_layer, dtype=tf.bfloat16)

        with tf.variable_scope('SimpleMultiTask',
                               reuse=tf.AUTO_REUSE):
            self.clk_model, self.clk_logits = self.__build_clk_model()
            self.buy_model, self.buy_logits = self.__build_buy_model()
            self.model = tf.squeeze(tf.stack([self.clk_model, self.buy_model], axis=1), [-1])

        with tf.name_scope('head'):
            self.train_op, self.loss = self.__compute_loss()
            self.acc, self.acc_op = tf.metrics.accuracy(labels=self.__label,
                                                        predictions=tf.round(self.model))
            self.auc, self.auc_op = tf.metrics.auc(labels=self.__label,
                                                   predictions=self.model,
                                                   num_thresholds=1000)
            tf.summary.scalar('eval_acc', self.acc)
            tf.summary.scalar('eval_auc', self.auc)

    def __create_dense_layer(self, input, num_hidden_units, activation, layer_name):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as mlp_layer_scope:
            dense_layer = tf.layers.dense(input,
                                       units=num_hidden_units,
                                       activation=activation,
                                       name=mlp_layer_scope)
            add_layer_summary(dense_layer, mlp_layer_scope.name)
        return dense_layer

    def __compute_loss(self):
        bce_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.model = tf.squeeze(self.model)
        loss = tf.math.reduce_mean(bce_loss_func(self.__label, self.model))
        tf.summary.scalar('loss', loss)

        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.__learning_rate)

        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op, loss

    def __build_clk_model(self):
        if self.__bf16:
            d_clk = tf.cast(self.__input_layer, dtype=tf.bfloat16)
            with tf.variable_scope('clk_model',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE).keep_weights():
                for layer_id, num_hidden_units in enumerate(self.__mlp):
                    d_clk = self.__create_dense_layer(d_clk, num_hidden_units, tf.nn.relu, f'd{layer_id}_clk')

                d_clk = self.__create_dense_layer(d_clk, 1, None, 'output_clk')
                d_clk = tf.cast(d_clk, tf.float32)
        else:
            with tf.variable_scope('clk_model',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                d_clk = self.__input_layer
                for layer_id, num_hidden_units in enumerate(self.__mlp):
                    d_clk = self.__create_dense_layer(d_clk, num_hidden_units, tf.nn.relu, f'd{layer_id}_clk')

                d_clk = self.__create_dense_layer(d_clk, 1, None, 'output_clk')
        Y_clk = tf.squeeze(d_clk)

        return tf.math.sigmoid(d_clk), Y_clk

    def __build_buy_model(self):
        if self.__bf16:
            d_buy = tf.cast(self.__input_layer, dtype=tf.bfloat16)
            with tf.variable_scope('buy_model',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE).keep_weights():
                d_buy = self.__input_layer
                for layer_id, num_hidden_units in enumerate(self.__mlp):
                    d_buy = self.__create_dense_layer(d_buy, num_hidden_units, tf.nn.relu, f'd{layer_id}_buy')

                d_buy = self.__create_dense_layer(d_buy, 1, None, 'output_buy')
            d_buy = tf.cast(d_buy, tf.float32)
        else:
            with tf.variable_scope('buy_model',
                                   partitioner=self.__dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                d_buy = self.__input_layer
                for layer_id, num_hidden_units in enumerate(self.__mlp):
                    d_buy = self.__create_dense_layer(d_buy, num_hidden_units, tf.nn.relu, f'd{layer_id}_buy')

                d_buy = self.__create_dense_layer(d_buy, 1, None, 'output_buy')

        Y_buy = tf.squeeze(d_buy)

        return tf.math.sigmoid(d_buy), Y_buy

def train(model,
          iterator,
          output_dir,
          train_dataset,
          test_dataset,
          keep_checkpoint_max,
          train_steps,
          test_steps=None,
          no_eval=False,
          timeline_steps=None,
          save_steps=None,
          checkpoint_dir=None,
          tf_config=None,
          server=None,
          inter=None,
          intra=None
          ):
    output_dir = os.path.join(output_dir, 'model_SimpleMultiTask_' + str(int(time.time())))
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

    if tf_config:
        sess_config = tf.ConfigProto()
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

        # train distributed model
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
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            writer = tf.summary.FileWriter(output_dir, sess.graph)
            merged = tf.summary.merge_all()
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.keep_checkpoint_max)
            run_metadata = tf.RunMetadata()
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

            # train model
            sess.run(train_init_op)
            start = time.perf_counter()
            for i in range(0, train_steps):
                if ((save_steps and save_steps > 0 and (i % save_steps == 0 or i == train_steps - 1))
                    or (checkpoint_dir and not save_steps and i == train_steps - 1)):
                    _, train_loss, events = sess.run([model.train_op, model.loss, merged])
                    writer.add_summary(events, i)
                    checkpoint_path = saver.save(sess,
                                                 save_path=os.path.join(checkpoint_dir, 'smt-checkpoint'),
                                                 global_step=i)
                    print(f'Save checkpoint to {checkpoint_path}')
                elif timeline_steps and timeline_steps > 0 and i % timeline_steps == 0:
                    _, train_loss = sess.run([model.train_op, model.loss],
                                             options=options,
                                             run_metadata=run_metadata)
                    fetched_timeline = tf_timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    print(f'Save timeline to {output_dir}')
                    with open(
                            os.path.join(output_dir,
                                         f'timeline-{i}.json'), 'w') as f:
                                         f.write(chrome_trace)
                else:
                    _, train_loss = sess.run([model.train_op, model.loss])

                # print training loss and time cost
                if i % 100 == 0 or i == train_steps - 1:
                    end = time.perf_counter()
                    cost_time = end - start
                    global_step_sec = (100 if i % 100 == 0 else train_steps - 1 % 100) / cost_time
                    print(f'global_step/sec: {global_step_sec:.4f}')
                    print(f'loss = {train_loss}, steps = {i}, cost time = {cost_time:0.2f}s')
                    start = time.perf_counter()

            # eval model
            if not no_eval:
                writer = tf.summary.FileWriter(os.path.join(args.output_dir, 'eval'))
                sess.run(tf.local_variables_initializer())
                sess.run(test_init_op)
                for i in range(test_steps - 1):
                    sess.run([model.acc, model.acc_op, model.auc, model.auc_op])
                _, eval_acc, _, eval_auc, events = sess.run([model.acc, model.acc_op, model.auc, model.auc_op, merged])
                writer.add_summary(events, i)
                print(f'Evaluation complete:[{i}/{test_steps}]')
                print(f'ACC = {eval_acc}\nAUC = {eval_auc}')

def parse_args():
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
                        help='not evaluate trained model by eval dataset.',
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    SEED = args.seed
    tf.set_random_seed(SEED)

    print('Validating arguments')
    train_file = os.path.join(args.data_location, 'taobao_train_data')
    test_file = os.path.join(args.data_location, 'taobao_test_data')
    if not os.path.exists(args.data_location):
        raise ValueError(f'[ERROR] data location: {args.data_location} does not exist. '
                         'Please provide valid path')

    if not os.path.exists(train_file) or (not args.no_eval and not os.path.exists(test_file)):
        raise ValueError('[ERROR] taobao_train_data or taobao_test_data does not exist '
                         'in the given data_location. Please provide valid path')

    # set params
    # set batch size & steps
    batch_size = args.batch_size
    no_of_training_examples = sum(1 for _ in open(train_file))
    if args.steps == 0:
        no_epochs = 10
        train_steps = math.ceil((float(no_epochs) * no_of_training_examples) / batch_size)
    else:
        no_epochs = math.ceil((float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps

    print(f'Numbers of training dataset: {no_of_training_examples}')
    print(f'Number of epochs: {no_epochs}')
    print(f'Number of train steps: {train_steps}')

    if not args.no_eval:
        no_of_test_examples = sum(1 for _ in open(test_file))
        test_steps = math.ceil(float(no_of_test_examples) / batch_size)
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

                train_dataset = generate_input_data(train_file, batch_size, no_epochs, SEED)
                test_dataset = generate_input_data(test_file, batch_size, 1, SEED)

                feature_columns = build_feature_cols()
                iterator = tf.data.Iterator.from_structure(tf.data.get_output_types(train_dataset),
                                                           tf.data.get_output_shapes(test_dataset))
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

                model = SimpleMultiTask(next_element,
                                        feature_columns,
                                        bf16=args.bf16,
                                        learning_rate=args.learning_rate,
                                        input_layer_partitioner=input_layer_partitioner,
                                        dense_layer_partitioner=dense_layer_partitioner)

                train(model,
                      iterator,
                      args.output_dir,
                      train_dataset,
                      test_dataset,
                      args.keep_checkpoint_max,
                      train_steps,
                      test_steps if not args.no_eval else None,
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

        train_dataset = generate_input_data(train_file, batch_size, no_epochs, SEED)
        test_dataset = generate_input_data(test_file, batch_size, 1, SEED)

        feature_columns = build_feature_cols()

        iterator = tf.data.Iterator.from_structure(tf.data.get_output_types(train_dataset),
                                                   tf.data.get_output_shapes(test_dataset))
        next_element = iterator.get_next()

        model = SimpleMultiTask(next_element,
                                feature_columns,
                                bf16=args.bf16,
                                learning_rate=args.learning_rate)
        train(model,
              iterator,
              args.output_dir,
              train_dataset,
              test_dataset,
              args.keep_checkpoint_max,
              train_steps,
              test_steps if not args.no_eval else None,
              no_eval=args.no_eval,
              timeline_steps=args.timeline,
              save_steps=args.save_steps,
              checkpoint_dir=args.checkpoint_dir)
