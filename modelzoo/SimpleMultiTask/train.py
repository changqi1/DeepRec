import tensorflow as tf
from tensorflow.python.client import timeline as tf_timeline
import collections
import os
import math
import time

LABEL_COLUMNS = ["clk", "buy"]
HASH_INPUTS = [
        "pid",
        "adgroup_id",
        "cate_id",
        "campaign_id",
        "customer",
        "brand",
        "user_id",
        "cms_segid",
        "cms_group_id",
        "final_gender_code",
        "age_level",
        "pvalue_level",
        "shopping_level",
        "occupation",
        "new_user_class_level",
        "tag_category_list",
        "tag_brand_list"
        ]
IDENTITY_INPUTS = ["price"]
HASH_BUCKET_SIZES = {
        "pid": 10,
        "adgroup_id": 100000,
        "cate_id": 10000,
        "campaign_id": 100000,
        "customer": 100000,
        "brand": 100000,
        "user_id": 100000,
        "cms_segid": 100,
        "cms_group_id": 100,
        "final_gender_code": 10,
        "age_level": 10,
        "pvalue_level": 10,
        "shopping_level": 10,
        "occupation": 10,
        "new_user_class_level": 10,
        "tag_category_list": 100000,
        "tag_brand_list": 100000
        }

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed",
            help="set random seed",
            type=int,
            default=2021)
    parser.add_argument("--data_location",
            help="Full path of train data",
            required=False,
            default="./data")
    parser.add_argument("--steps",
            help="set the number of steps on train dataset",
            type=int,
            default=0)
    parser.add_argument("--batch_size",
            help="Batch size to train",
            type=int,
            default=512)
    parser.add_argument("--timeline",
            help="number of steps on saving timeline",
            type=int,
            default=0)
    parser.add_argument("--output_dir",
            help="Full path to logs & model output directory",
            required=False,
            default="./result")
    parser.add_argument("--no_eval",
            help="not evaluate trained model by eval dataset.",
            action="store_true")
    parser.add_argument("--save_steps",
            help="set the number of steps on saving checkpoints",
            type=int,
            default=0)
    parser.add_argument("--keep_checkpoint_max",
            help="Maximum number of recent checkpoint to keep",
            type=int,
            default=1)
    parser.add_argument("--bf16",
            help="enable DeepRec BF16 in deep model",
            action="store_true")
    return parser.parse_args()

def exponential_decay_with_burnin(global_step,
        learning_rate_base,
        learning_rate_decay_steps,
        learning_rate_decay_factor,
        burnin_learning_rate=0.0,
        burnin_steps=0,
        min_learning_rate=0.0,
        staircase=True):
    if burnin_learning_rate == 0:
        burnin_rate = learning_rate_base
    else:
        slope = (learning_rate_base - burnin_learning_rate) / burnin_steps
        burnin_rate = slope * tf.cast(global_step, tf.float32) + burnin_learning_rate
    post_burnin_learning_rate = tf.train.exponential_decay(
            learning_rate_base,
            global_step - burnin_steps,
            learning_rate_decay_steps,
            learning_rate_decay_factor,
            staircase=staircase)
    return tf.maximum(
            tf.where(
                tf.less(tf.cast(global_step, tf.int32), tf.constant(burnin_steps)),
                burnin_rate, post_burnin_learning_rate),
            min_learning_rate,
            name='learning_rate')

args = parse_args()

tf.set_random_seed(args.seed)

train_data_file = os.path.join(args.data_location, "taobao_train_data")
test_data_file = os.path.join(args.data_location, "taobao_test_data")
train_file = open(train_data_file)
no_of_training_examples = sum(1 for _ in train_file)
train_file.close()
if args.steps == 0:
    no_epochs = 10
    train_steps = math.ceil((float(no_epochs) * no_of_training_examples) / args.batch_size)
else:
    no_epochs = math.ceil((float(args.batch_size) * args.steps) / no_of_training_examples)
    train_steps = args.steps

defaults = [[0]] * len(LABEL_COLUMNS) + [[" "]] * len(HASH_INPUTS) + [[0]] * len(IDENTITY_INPUTS)
headers = LABEL_COLUMNS + HASH_INPUTS + IDENTITY_INPUTS

def parse_csv_and_split_on_labels_features_tuple(x):
    l = list(zip(headers, tf.io.decode_csv(x, defaults)))
    # This is because Dataset.map() have strange requirement of using collections.OrderedDict
    # otherwise throws type exception.
    return collections.OrderedDict(l[:2]), collections.OrderedDict(l[2:])

train_dataset = (
        tf.data.TextLineDataset(train_data_file)
        .shuffle(buffer_size=50000, seed=args.seed)
        .repeat(no_epochs)
        .batch(args.batch_size)
        .map(parse_csv_and_split_on_labels_features_tuple)
        )

test_dataset = (
        tf.data.TextLineDataset(test_data_file)
        .shuffle(buffer_size=50000, seed=args.seed)
        .repeat(no_epochs)
        .batch(args.batch_size)
        .map(parse_csv_and_split_on_labels_features_tuple)
        )

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
                combiner="mean"))

for i in range(len(IDENTITY_INPUTS)):
    tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(IDENTITY_INPUTS[i], 50),
            dimension=16,
            combiner="mean")

iterator = tf.data.Iterator.from_structure(tf.data.get_output_types(train_dataset), tf.data.get_output_shapes(test_dataset))
next_element = iterator.get_next()
train_initialization_operation = iterator.make_initializer(train_dataset)
test_initialization_operation = iterator.make_initializer(test_dataset)

with tf.variable_scope("SimpleMultiTask").keep_weights(dtype=tf.float32):
    X = tf.feature_column.input_layer(next_element[1], feature_columns)
    if args.bf16:
        X = tf.cast(X, dtype=tf.bfloat16)

    d1_clk = tf.layers.dense(X, units=256, activation=tf.nn.relu, name="d1_clk")
    d2_clk = tf.layers.dense(d1_clk, units=196, activation=tf.nn.relu, name="d2_clk")
    d3_clk = tf.layers.dense(d2_clk, units=128, activation=tf.nn.relu, name="d3_clk")
    d4_clk = tf.layers.dense(d3_clk, units=64, activation=tf.nn.relu, name="d4_clk")
    d5_clk = tf.layers.dense(d4_clk, units=1, name="output_clk")
    if args.bf16:
        d5_clk = tf.cast(d5_clk, tf.float32)
    Y_clk = tf.squeeze(d5_clk)
    loss_clk = tf.losses.sigmoid_cross_entropy(multi_class_labels=next_element[0]["clk"], logits=Y_clk)

    d1_buy = tf.layers.dense(X, units=256, activation=tf.nn.relu, name="d1_buy")
    d2_buy = tf.layers.dense(d1_buy, units=196, activation=tf.nn.relu, name="d2_buy")
    d3_buy = tf.layers.dense(d2_buy, units=128, activation=tf.nn.relu, name="d3_buy")
    d4_buy = tf.layers.dense(d3_buy, units=64, activation=tf.nn.relu, name="d4_buy")
    d5_buy = tf.layers.dense(d4_buy, units=1, name="output_buy")
    if args.bf16:
        d5_buy = tf.cast(d5_buy, tf.float32)
    Y_buy = tf.squeeze(d5_buy)
    loss_buy = tf.losses.sigmoid_cross_entropy(multi_class_labels=next_element[0]["buy"], logits=Y_buy)

    total_loss = loss_buy + loss_clk
    learning_rate = exponential_decay_with_burnin(
            tf.train.get_or_create_global_step(),
            0.001,
            1000,
            0.5,
            min_learning_rate=1e-07)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=tf.train.get_or_create_global_step())
    tf.summary.scalar("loss", total_loss)

    preds = tf.squeeze(tf.stack([tf.math.sigmoid(d5_clk), tf.math.sigmoid(d5_buy)], axis=1), [-1])
    labels = tf.stack([next_element[0]["clk"], next_element[0]["buy"]], axis=1)

    acc, acc_op = tf.metrics.accuracy(
            labels,
            preds,
            name="acc")
    auc, auc_op = tf.metrics.auc(
            labels,
            preds,
            name = "auc")
    tf.summary.scalar("acc", acc)
    tf.summary.scalar("auc", auc)

with tf.Session() as sess:
    writer = tf.summary.FileWriter(args.output_dir, sess.graph)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables(),
            max_to_keep=args.keep_checkpoint_max)
    run_metadata = tf.RunMetadata()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(train_initialization_operation)
    start = time.perf_counter()
    for i in range(0, train_steps):
        if args.save_steps > 0 and (i % args.save_steps == 0
                or i == train_steps - 1):
            _, train_loss, events = sess.run(
                    [train_op, total_loss, merged])
            writer.add_summary(events, i)
            checkpoint_path = saver.save(sess,
                    save_path=os.path.join(
                        args.output_dir,
                        "smt-checkpoint"),
                    global_step=i)
            print("Save checkpoint to %s" % checkpoint_path)
        elif args.timeline > 0 and i % args.timeline == 0:
            _, train_loss = sess.run([train_op, total_loss],
                    options=options,
                    run_metadata=run_metadata)
            fetched_timeline = tf_timeline.Timeline(
                    run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format(
                    )
            print("Save timeline to %s" % args.output_dir)
            with open(
                    os.path.join(args.output_dir,
                        "timeline-%d.json" % i), "w") as f:
                        f.write(chrome_trace)
        else:
            _, train_loss = sess.run([train_op, total_loss])

        if i % 100 == 0 or i == train_steps - 1:
            end = time.perf_counter()
            cost_time = end - start
            global_step_sec = (100 if i % 100 == 0 else train_steps - 1 % 100) / cost_time
            print("global_step/sec: %0.4f" % global_step_sec)
            print("loss = {}, steps = {}, cost time = {:0.2f}s".format(
                train_loss, i, cost_time))
            start = time.perf_counter()
    if not args.no_eval:
        writer = tf.summary.FileWriter(
                os.path.join(args.output_dir, "eval"))
        sess.run(tf.local_variables_initializer())
        sess.run(test_initialization_operation)
        test_file = open(test_data_file)
        no_of_test_examples = sum(1 for _ in test_file)
        test_file.close()
        test_steps = math.ceil(float(no_of_test_examples) / args.batch_size)
        for i in range(test_steps - 1):
            sess.run([acc, acc_op, auc, auc_op])
        _, eval_acc, _, eval_auc, events = sess.run([acc, acc_op, auc, auc_op, merged])
        writer.add_summary(events, i)
        print("Evaluation complate:[{}/{}]".format(i, test_steps))
        print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))
