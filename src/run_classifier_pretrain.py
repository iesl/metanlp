
# coding=utf-8
# Copyright Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import utils_impl as saved_model_utils
import re


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir",
    "",
    "The input data dir. Should contain the tf_records files (or other data files) "
    "for the tasks.")

flags.DEFINE_string(
    "supervised_data_dir",
    "",
    "The input data dir for supervised data. Should contain the tf_records files (or other data files) "
    "for the tasks.")

flags.DEFINE_string(
    "bert_config_file",
    "",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_train_files",
                    "",
                    "The tf_record train files for all tasks.")

flags.DEFINE_string("task_eval_files",
                    "",
                    "The tf_record train files for all tasks.")

flags.DEFINE_string("vocab_file",
                    "pretrained-model/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", 'output',
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint",
    "pretrained-model/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("num_train_tasks", 6, "Total tasks in training.")
flags.DEFINE_integer("num_eval_tasks", 6, "Total validation tasks during training.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("train_lr", 5e-5, "The learning rate for MAML updates.")

flags.DEFINE_integer("meta_batchsz", 8, "The batch-size for each task.")

flags.DEFINE_integer("SGD_K", 2, "The initial learning rate for Adam.")

flags.DEFINE_bool("stop_grad", True, "Whether to use first order approximation")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# tf.flags.DEFINE_string(
#     "tpu_name", None,
#     "The Cloud TPU to use for training. This should be either the name "
#     "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#     "url.")
#
# tf.flags.DEFINE_string(
#     "tpu_zone", None,
#     "[Optional] GCE zone where the Cloud TPU is located in. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")
#
# tf.flags.DEFINE_string(
#     "gcp_project", None,
#     "[Optional] Project name for the Cloud TPU-enabled project. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

# flags.DEFINE_integer(
#     "num_tpu_cores", 8,
#     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float(
    "keep_prob", 0.9,
    "dropout keep prob")

flags.DEFINE_bool("use_pooled_output", True, "Whether to use pooled output, if false use weighted all layer")

flags.DEFINE_string("warm_start_model_dir", None, "Saved Model Dir")

flags.DEFINE_integer(
    "min_layer_with_grad", 0,
    "Minimum bert layer to adapt based on task")

flags.DEFINE_bool(
    "train_word_embeddings", False, "whether to train word embeddings"
)

flags.DEFINE_float(
    "attention_probs_dropout_prob", None, "Dropout for attention in bert"
)

flags.DEFINE_float(
    "hidden_dropout_prob", None, "Dropout for hidden layer in bert"
)

flags.DEFINE_bool(
    "debug", False, "debug mode"
)

flags.DEFINE_integer("tasks_per_gpu", 6, "Tasks to keep on one gpu")

flags.DEFINE_bool(
    "adapt_layer_norm", True, "whether to adapt layer norm parameters"
)

flags.DEFINE_integer("output_layers", 1, "Number of layers in output MLP")

flags.DEFINE_string("tasks_labels", "2,2", "Num classes in each task (comma-separated)")

flags.DEFINE_integer("label_emb_size", 256, "Label and output encoding size")

flags.DEFINE_integer("num_batches", 0, "Number of batches for continuous training.")

flags.DEFINE_bool("use_euclidean_norm", True, "Use euclidean norm for computing distance from label embeddings")

flags.DEFINE_bool("update_only_label_embedding", False, "Update only label embedding in inner loop")

flags.DEFINE_bool("is_meta_sgd", True, "Learn per-layer learning rates")

flags.DEFINE_bool("use_exp_lr", False, "Use learing-rate with exponential transform for meta-sgd")

flags.DEFINE_integer("test_num_labels", 2, "num labels in testing")

flags.DEFINE_integer("eval_examples_per_task", 2000, "examples used for validation per task")

flags.DEFINE_bool("prototypical_baseline", False, "use prototypical baseline: dont share encoders")

flags.DEFINE_bool("data_sqrt_sampling", False, "whether to sample from square root of ")

flags.DEFINE_integer("deep_set_layers", 0, "number of layers of label_emb projections")

flags.DEFINE_string("activation_fn", "tanh",
                    "activation to use in output layers and label layers: tanh, relu, gelu, linear")

flags.DEFINE_integer("num_gpus", 4, "number of gpus for training")

flags.DEFINE_bool("clip_lr", False, "whether to clip lr to minimum of 1e-8 ")

flags.DEFINE_integer("inner_epochs", 1, "number of epochs for inner loop")

flags.DEFINE_bool("randomize_inner_steps", False, "whether to randomize number of inner loop steps")

flags.DEFINE_integer("min_inner_steps", 3, "minimum number of inner loop steps in one epoch")

flags.DEFINE_bool("warp_layers", False, "whether to treat intermediate MLP as warps")

flags.DEFINE_bool("average_query_loss", False, "whether to use average of per-step query loss")

flags.DEFINE_integer("average_query_every", 1, "average of per-step query loss after this many steps")

flags.DEFINE_bool("sgd_first_batch", False, "whether to use the first batch (used for softmax) for adaptation")

flags.DEFINE_float("pretrain_task_weight", 1.0, "total weight on pretrain tasks for multi-task learning")

flags.DEFINE_bool("weight_query_loss", False, "whether to weight the per step query loss")

flags.DEFINE_integer("max_train_batch_size", -1, "override train_batch_size with this")

flags.DEFINE_bool("decay_finetune_lr", True, "decay finetune learning rate")

flags.DEFINE_bool("fix_bert", False, "fix BERT model in outer loop")


def input_fn_builder(input_files, seq_length, is_training, drop_remainder, nexamples_per_file=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = {
      # "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }
  if nexamples_per_file is None:
      nexamples_per_file = [1] * len(input_files)

  _label_map = {'mnli': 3, 'sst': 2, 'mrpc': 2, 'winograd': 2, 'rte': 2, 'snli': 3, 'quora': 2,
                'acceptability': 2, 'sts': 5, 'amazon': 2, 'squad': 2,
                'asv': 3, 'hst': 3, 'svt': 3, 'ahs': 3, 'aht': 3, 'hsv': 3,
                'fewrel': 5
                }

  def _decode_record(record, name_to_features, nlabels):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    example['num_labels'] = nlabels
    return example

  def input_fn(params):
    """The actual input function."""
    if is_training:
        batch_size = params["train_batch_size"]
    else:
        batch_size = params["eval_batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if FLAGS.do_train:
        if is_training:
            input_labels = []
            for i, fname in enumerate(input_files):
                if fname.split("/")[-1].startswith('meta_pretrain'):
                    input_labels.append(int(fname.split("/")[-1].split(".tf_record")[0].split("_")[-1]))
                elif fname.split("/")[-1] in ['mnli_train.tf_record', 'snli_train.tf_record', 'sst_train.tf_record',
                                              'asv_c_eval.tf_record', 'hsv_c_eval.tf_record', 'svt_c_eval.tf_record',
                                              'ahs_c_eval.tf_record', 'aht_c_eval.tf_record']:
                    input_labels.append(_label_map[fname.split("/")[-1].split("_")[0]])
                else:
                    input_labels.append(2)
                tf.logging.info('File: %s, nlabels: %d' % (fname, input_labels[i]))
        else:
            input_labels = [_label_map[fname.split("/")[-1].split("_")[0]] if fname.split("/")[-1].split("_")[0] in _label_map else 2
                            for fname in input_files]
        input_files_and_labels = list(zip(input_files, input_labels, nexamples_per_file))

    datasets = []
    pretrain_weight_ids = []
    sampling_weights = []
    if FLAGS.do_train:
        for f_id, (fname, nlabels, nexamples) in enumerate(input_files_and_labels):
            # nlabels = 3 if is_training else nlabels
            dataset = tf.data.TFRecordDataset(fname).map(lambda x: _decode_record(x, name_to_features, nlabels))
            if is_training:
                # This is training
                if not fname.split("/")[-1].startswith('meta_pretrain'):
                    dataset = dataset.shuffle(buffer_size=1000)
                if FLAGS.data_sqrt_sampling:
                    tf.logging.info('Using sqrt-sampling')
                    sampling_weights.append(np.sqrt(nexamples))
                else:
                    sampling_weights.append(1.0)
                if fname.split("/")[-1].startswith('meta_pretrain'):
                    pretrain_weight_ids.append(f_id)
            else:
                # This is validation
                dataset = dataset.repeat().shuffle(buffer_size=500)
                dataset = dataset.take(FLAGS.eval_examples_per_task)
            if fname.split("/")[-1].startswith('meta_pretrain') or FLAGS.max_train_batch_size < 0:
                dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
            else:
                dataset = dataset.batch(FLAGS.max_train_batch_size, drop_remainder=drop_remainder)
            if is_training:
                dataset = dataset.repeat()
            datasets.append(dataset)

        if is_training:
            if FLAGS.pretrain_task_weight < 1.0:
                pretrain_weight_ids = set(pretrain_weight_ids)
                total_p_weight = sum([wt for d_id, wt in enumerate(sampling_weights) if d_id in pretrain_weight_ids])
                total_other_weight = sum(sampling_weights) - total_p_weight
                for d_id, wt in enumerate(sampling_weights):
                    if d_id in pretrain_weight_ids:
                        sampling_weights[d_id] = FLAGS.pretrain_task_weight * wt / total_p_weight
                    else:
                        sampling_weights[d_id] = (1.0 - FLAGS.pretrain_task_weight) * wt / total_other_weight
            else:
                sampling_weights = [w / sum(sampling_weights) for w in sampling_weights]
            tf.logging.info('Sampling weights for input_files %s : %s' % (", ".join(input_files), ", ".join(map(str, sampling_weights))))
            sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, sampling_weights)
            sampled_dataset = sampled_dataset.batch(FLAGS.tasks_per_gpu, drop_remainder=drop_remainder)
        else:
            sampled_dataset = datasets[0]
            for data in datasets[1:]:
                sampled_dataset = sampled_dataset.concatenate(data)
            sampled_dataset = sampled_dataset.batch(FLAGS.tasks_per_gpu, drop_remainder=drop_remainder)

        return sampled_dataset
    elif FLAGS.do_eval:
        if is_training:
            nlabels = FLAGS.test_num_labels
            dataset = tf.data.TFRecordDataset(input_files[0]).map(lambda x: _decode_record(x, name_to_features, nlabels))
            first_batch_dataset = dataset.shuffle(buffer_size=100).take(batch_size)
            rest_dataset = dataset.repeat(int(FLAGS.num_train_epochs)).shuffle(buffer_size=500)
            dataset = first_batch_dataset.concatenate(rest_dataset)
            dataset = dataset.batch(batch_size)
        else:
            nlabels = FLAGS.test_num_labels
            dataset = tf.data.TFRecordDataset(input_files[1]).map(lambda x: _decode_record(x, name_to_features, nlabels))
            dataset = dataset.batch(batch_size)
        return dataset

  return input_fn


def convert_to_tensor(g):
    if isinstance(g, tf.IndexedSlices):
        return tf.convert_to_tensor(g)
    else:
        return g


def create_finetune_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                          labels, use_one_hot_embeddings, num_labels, num_train_steps = 1,
                          num_warmup_steps=10):
    """
       input_ids: (tasks, batchsize, max_seq_len)
       input_mask: (tasks, batchsize, max_seq_len)
       segment_ids: (tasks, batchsize, max_seq_len)
       labels:(tasks, batchsize)
    """
    task_weights, task_learning_rates, bert_learning_rates = create_task_weights_and_lr(bert_config,
                                                                                        FLAGS.label_emb_size)
    tf.logging.info('NUM TRAIN STEPS:: %d' %(num_train_steps))
    #import ipdb; ipdb.set_trace();
    bert_model = modeling.BertModel(
        config=bert_config,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert")

    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int64)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    def get_learning_rate(key, sgd_id, step=None):
        lr = determine_learning_rate(bert_learning_rates, task_learning_rates, key, sgd_id)

        steps_float = tf.cast(step, tf.float32)

        warmup_percent_done = steps_float / tf.maximum(1.0, warmup_steps_float)
        warmup_learning_rate = lr * warmup_percent_done

        if FLAGS.decay_finetune_lr:
            end_lr = 0.
            decayed_learning_rate = (lr - end_lr) * (1. - steps_float / num_train_steps) + end_lr
        else:
            decayed_learning_rate = lr

        is_warmup = tf.cast(step < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * decayed_learning_rate + is_warmup * warmup_learning_rate)
        return learning_rate

    #fast_task_weights = {k: v for k, v in task_weights}
    fast_task_weights = task_weights

    fast_weights = filter_by_layer(bert_model.weights)
    #fast_weights = {k: v for k,v in weights_for_grad}

    fast_weights_forward = {key: fast_weights[key] if key in fast_weights else bert_model.weights[key]
                            for key in bert_model.weights}
    label_mask = tf.ones((num_labels[0]))
    # if FLAGS.update_only_label_embedding or not is_training:
    with tf.variable_scope('new_variable'):
        label_emb_size = FLAGS.label_emb_size
        if not FLAGS.use_euclidean_norm:
            label_emb_size += 1
        label_embs = tf.get_variable("label_embedding",
                                     dtype=tf.float32, shape=(FLAGS.test_num_labels, label_emb_size),
                                     initializer=tf.zeros_initializer())

    if not is_training:
        support = (input_ids, input_mask, segment_ids, labels)
        support_loss, support_logits, support_acc, label_embs, _ = forward(support, bert_model,
                is_training, num_labels[0],
                weights=fast_weights_forward,
                label_embs=label_embs,
                label_mask=label_mask,
                task_weights=fast_task_weights,
                reuse=True)
        return support_loss, support_logits, support_acc, label_embs

    step = tf.get_variable('training_step', dtype=np.int64, shape=(), initializer=tf.constant_initializer(0),
                           trainable=False)

    support = (input_ids, input_mask, segment_ids, labels)
    
    sloss, _, _, label_embs_updated, _ = forward(support, bert_model, is_training, num_labels[0],
                                                    weights=fast_weights_forward,
                                                    task_weights=fast_task_weights,
                                                    label_embs=None,
                                                    label_mask=label_mask,
                                                    reuse=True,
                                                    create_label_emb=True)
    is_first_step = tf.cast(tf.equal(step, tf.constant(0, dtype=tf.int64)), tf.float32)
    label_embs = tf.assign(label_embs, label_embs + is_first_step * label_embs_updated)
    if not FLAGS.update_only_label_embedding and FLAGS.prototypical_baseline:
        train_op = [label_embs]
        return sloss, sloss, train_op
    support_loss, _, _, _, _ = forward(support, bert_model, is_training, num_labels[0],
                                weights=fast_weights_forward,
                                task_weights=fast_task_weights,
                                label_embs=label_embs,
                                label_mask=label_mask,
                                reuse=True,
                                create_label_emb=True)
    if not FLAGS.sgd_first_batch:
        support_loss = support_loss * (1 - is_first_step)

    if FLAGS.SGD_K > 1:
        sgd_k_id = (step * 5)/ int(num_train_steps)
    else:
        sgd_k_id = 0
    sgd_k_id = tf.cast(sgd_k_id, tf.int32)
    # print_op = tf.print('label_embs', {'label_embs': label_embs, 'step' : sgd_k_id, 'lr': get_learning_rate('label_embs', sgd_k_id, step)})
    if not FLAGS.use_pooled_output:
        p = {'bert_layer_weights_%d' % i : task_weights['bert_layer_weights'][i] for i in range(bert_config.num_hidden_layers)}
        p["step"] = step
        p["lr"] = get_learning_rate('label_embs', sgd_k_id, step)
        print_op = tf.print('layer weights', p)
    # with tf.control_dependencies([print_op]):
    grads = tf.gradients(support_loss, list(fast_weights.values()))
    if FLAGS.update_only_label_embedding:
        fast_task_weights = {key: value for key, value in fast_task_weights.items() if "label" not in key}
    task_grads = tf.gradients(support_loss, list(fast_task_weights.values()))
    all_grads = grads + task_grads
    if FLAGS.update_only_label_embedding:
        label_emb_grads = tf.gradients(support_loss, label_embs)
        # label_emb_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in label_emb_grads]
        all_grads += label_emb_grads

    new_grads = []
    for g in all_grads:
        if g is None:
            new_grads.append(g)
        elif isinstance(g, tf.IndexedSlices):
            new_grads.append(
                tf.IndexedSlices(tf.stop_gradient(g.values), g.indices, dense_shape=g.dense_shape))
        else:
            new_grads.append(tf.stop_gradient(g))
    all_grads = new_grads

    all_grads, _ = tf.clip_by_global_norm(all_grads, clip_norm=1.0)

    grads = all_grads[:len(grads)]
    task_grads = all_grads[len(grads):len(grads) + len(task_grads)]
    label_emb_grads = None
    if FLAGS.update_only_label_embedding:
        label_emb_grads = all_grads[len(grads) + len(task_grads):]

    gvs = dict(zip(fast_weights.keys(), grads))
    task_gvs = dict(zip(fast_task_weights.keys(), task_grads))
    # update theta_pi according to varibles
    fast_weights = dict(zip(fast_weights.keys(), [tf.assign(fast_weights[key], fast_weights[key] - get_learning_rate(key, sgd_k_id, step) * convert_to_tensor(gvs[key])
                                                    if gvs[key] is not None else fast_weights[key]) for key in fast_weights.keys()]))

    fast_task_weights = {key: tf.assign(fast_task_weights[key],
                                        fast_task_weights[key] - get_learning_rate(key, sgd_k_id, step) *
                                        convert_to_tensor(task_gvs[key]))
                            for key in fast_task_weights.keys()}
    if FLAGS.update_only_label_embedding:
        # import ipdb; ipdb.set_trace()
        label_embs = tf.assign(label_embs, label_embs - get_learning_rate('label_embs', sgd_k_id, step) * label_emb_grads[0])
    fast_weights_forward = {key: fast_weights[key] if key in fast_weights else bert_model.weights[key]
                            for key in bert_model.weights}
    # average loss
    total_loss = tf.reduce_mean(support_loss)
    all_sgd_ops = list(fast_task_weights.values()) + list(fast_weights.values())
    with tf.control_dependencies(all_sgd_ops):
        step_incr = tf.assign(step, step + 1)
    all_ops = all_sgd_ops + [step_incr]

    if FLAGS.update_only_label_embedding:
        all_ops += [label_embs]
    train_op = tf.group(all_ops)

    return total_loss, support_loss, train_op


def filter_by_layer(weights):
    ''' Return weights to train '''
    filtered_weights = {}
    for key, value in weights.items():
        layer_num = re.findall(r'layer_([0-9]+)', key)
        if not FLAGS.warp_layers and FLAGS.adapt_layer_norm and len(re.findall('LayerNorm', key)) != 0:
            filtered_weights[key] = value
        elif len(layer_num) > 0:
            layer_num = int(layer_num[0])
            if FLAGS.warp_layers and (len(re.findall('layer_%d/intermediate' % layer_num, key)) > 0 or
                                      len(re.findall('layer_%d/output' % layer_num, key)) > 0):
                # do not adapt intermediate MLP
                tf.logging.info('Not adapting %s' % key)
            elif layer_num >= FLAGS.min_layer_with_grad:
                # if > min_layer then we train
                filtered_weights[key] = value
        else:
            if len(re.findall('embeddings', key)) == 0 or FLAGS.train_word_embeddings:
                # if not embedding layer then train
                filtered_weights[key] = value

    return filtered_weights


def forward(inputs, model, is_training, num_labels,
            weights, task_weights=None, reuse=False, label_embs=None, label_mask=None,
            return_per_example_loss=False, create_label_emb=False):
    _ids_x, _mask_x, _segment_ids_x, _labels = inputs
    model.forward(
        weights=weights,
        is_training=is_training,
        input_ids=_ids_x,
        input_mask=_mask_x,
        token_type_ids=_segment_ids_x,
        reuse=reuse)

    if not FLAGS.use_pooled_output:
        all_bert_layers = model.get_all_encoder_layers()
        bert_layer_weights = task_weights['bert_layer_weights']
        if FLAGS.do_eval:
            bert_layer_weights = tf.reshape(tf.nn.softmax(bert_layer_weights), [1, 1, -1])
        hidden_reps = tf.reduce_sum(
            tf.stack(all_bert_layers, axis=-1)[:, 0, :, :] * bert_layer_weights, -1)
        # (tasks * batchsize, emb_dim)
        # output_layer = hidden_reps[:, 0, :]
        output_layer = tf.nn.tanh(tf.matmul(hidden_reps, task_weights['bert_proj_weight']))
    else:
        output_layer = model.get_pooled_output()
    # hidden_size = output_layer.shape[-1].value
    if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=FLAGS.keep_prob)

    output_logits = output_layer
    for wid in range(FLAGS.output_layers):
        output_logits = tf.matmul(output_logits, task_weights['output_weights_%d' % wid], transpose_b=True)
        output_logits = tf.nn.bias_add(output_logits, task_weights['output_bias_%d' % wid])
        if wid < FLAGS.output_layers - 1:
            # output_logits = tf.nn.tanh(output_logits)
            output_logits = modeling.get_activation(FLAGS.activation_fn)(output_logits)

    if label_embs is None or create_label_emb:
        label_logits = output_layer
        for wid in range(FLAGS.output_layers):
            label_logits = tf.matmul(label_logits, task_weights['label_weights_%d' % wid], transpose_b=True)
            label_logits = tf.nn.bias_add(label_logits, task_weights['label_bias_%d' % wid])
            if wid < FLAGS.output_layers - 1:
                # label_logits = tf.nn.tanh(label_logits)
                label_logits = modeling.get_activation(FLAGS.activation_fn)(label_logits)

        # unique_labels, mapped_labels, count_labels = tf.unique_with_counts(_labels)
        # label_embs_shape = tf.concat([tf.shape(unique_labels), tf.shape(label_logits)[-1]], axis=-1)
        # label_embs_shape = tf.concat([[num_labels], [tf.shape(label_logits, out_type=tf.int64)[-1]]], axis=-1)
        label_embs_shape = [num_labels, modeling.get_shape_list(label_logits)[-1]]
        _labels_2d = tf.expand_dims(_labels, -1)

        label_embs_new = tf.scatter_nd(indices=_labels_2d, updates=label_logits, shape=label_embs_shape)
        # import ipdb; ipdb.set_trace()
        one_hot_labels = tf.one_hot(_labels, depth=num_labels, dtype=tf.float32)
        # import ipdb; ipdb.set_trace()
        label_counts = tf.transpose(tf.reduce_sum(one_hot_labels, 0))
        label_embs_new = label_embs_new / tf.maximum(tf.cast(tf.expand_dims(label_counts, -1), tf.float32), 1.0)
        for lid in range(FLAGS.deep_set_layers):
            label_embs_new = tf.matmul(label_embs_new, task_weights['label_set_weights_%d' % lid])
            label_embs_new = tf.nn.bias_add(label_embs_new, task_weights['label_set_bias_%d' % lid])
            if lid < FLAGS.deep_set_layers - 1:
                label_embs_new = modeling.get_activation(FLAGS.activation_fn)(label_embs_new)
        if label_embs is None:
            label_embs = label_embs_new
        if label_mask is None:
            label_mask = tf.cast(tf.greater(label_counts, 0), tf.float32)
    else:
        label_embs_new = label_embs

    loss, accuracy, logits = task_loss(output_logits, label_embs, _labels, label_mask, num_labels)
    return loss, logits, accuracy, label_embs_new, label_mask


def task_loss(features, label_embs, labels, label_mask, num_labels):
    with tf.variable_scope("task_loss"):
        if FLAGS.use_euclidean_norm:
            # import ipdb; ipdb.set_trace()
            logits = -tf.reduce_sum(tf.math.squared_difference(tf.expand_dims(features, -2), tf.expand_dims(label_embs, 0)), -1)
        else:
            # label_emb_norm = tf.norm(label_embs, axis=-1, keep_dims=True) + 1e-6
            # label_embs = label_embs / label_emb_norm
            _, _edim = label_embs.get_shape().as_list()
            tf.logging.info('Using as softmax parameters, label_emb_dim: %d' % (_edim))
            logits = tf.matmul(features, label_embs[:, :FLAGS.label_emb_size], transpose_b=True)
            logits = tf.nn.bias_add(logits, label_embs[:, -1])
        if FLAGS.stop_grad:
            per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        else:
            # probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

        # example_mask = tf.scatter_nd(indices=labels, updates=label_mask, shape=tf.shape(labels))
        example_mask = tf.gather(indices=labels, params=label_mask)
        per_example_loss = per_example_loss  # * example_mask
        accuracy = tf.contrib.metrics.accuracy(tf.argmax(logits, axis=1, output_type=tf.int64), labels)
        loss = tf.reduce_mean(per_example_loss)
        # loss = tf.reduce_sum(per_example_loss * example_mask) / tf.maximum(tf.reduce_sum(example_mask), 1.0)
    return loss, accuracy, logits


def create_task_weights_and_lr(bert_config, noutput):
    nhidden = bert_config.hidden_size
    task_weights = {}
    learning_rates = {}
    train_lr = FLAGS.train_lr
    if FLAGS.use_exp_lr:
        train_lr = np.log(train_lr)
    for wid in range(FLAGS.output_layers):
        outnhidden = nhidden // 2 if wid < FLAGS.output_layers - 1 else noutput
        output_weights = tf.get_variable(
            "output_weights_%d" % wid, [outnhidden, nhidden],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "output_bias_%d" % wid, [outnhidden], initializer=tf.zeros_initializer())
        task_weights['output_weights_%d' % wid] = output_weights
        task_weights['output_bias_%d' % wid] = output_bias
        learning_rates['output/layer%d' % wid] = []
        for sid in range(FLAGS.SGD_K):
            with tf.variable_scope('sgd%d' % sid):
                if FLAGS.warp_layers:
                    lr = tf.get_variable("output/layer%d/learning_rate" % wid, dtype=tf.float32, shape=(),
                                         initializer=tf.constant_initializer(0.),
                                         trainable=False)
                else:
                    lr = tf.get_variable("output/layer%d/learning_rate" % wid, dtype=tf.float32, shape=(),
                                         initializer=tf.constant_initializer(train_lr),
                                         trainable=FLAGS.is_meta_sgd,
                                         constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
                learning_rates['output/layer%d' % wid].append(lr)
                tf.summary.scalar('output_layer%d_sgd%d' % (wid, sid), lr)
        nhidden = outnhidden

    nhidden = bert_config.hidden_size
    for wid in range(FLAGS.output_layers):
        outnhidden = nhidden // 2 if wid < FLAGS.output_layers - 1 else noutput
        if not FLAGS.use_euclidean_norm and wid == FLAGS.output_layers - 1:
            tf.logging.info('Not using euclidean norm')
            outnhidden = noutput + 1
        if FLAGS.prototypical_baseline:
            tf.logging.info('Using Prototypical Baseline')
            task_weights['label_weights_%d' % wid] = task_weights['output_weights_%d' % wid]
            task_weights['label_bias_%d' % wid] = task_weights['output_bias_%d' % wid]
        else:
            output_weights = tf.get_variable(
                "label_weights_%d" % wid, [outnhidden, nhidden],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias = tf.get_variable(
                "label_bias_%d" % wid, [outnhidden], initializer=tf.zeros_initializer())
            task_weights['label_weights_%d' % wid] = output_weights
            task_weights['label_bias_%d' % wid] = output_bias
        learning_rates['label/layer%d' % wid] = []
        for sid in range(FLAGS.SGD_K):
            with tf.variable_scope('sgd%d' % sid):
                if FLAGS.prototypical_baseline:
                    learning_rates['label/layer%d' % wid].append(learning_rates['output/layer%d' % wid][sid])
                else:
                    lr = tf.get_variable("label/layer%d/learning_rate" % wid, dtype=tf.float32, shape=(),
                                         initializer=tf.constant_initializer(train_lr),
                                         trainable=FLAGS.is_meta_sgd,
                                         constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
                    learning_rates['label/layer%d' % wid].append(lr)
                    tf.summary.scalar('label_layer%d_sgd%d' % (wid, sid), lr)
        nhidden = outnhidden

    for wid in range(FLAGS.deep_set_layers):
        output_weights = tf.get_variable(
            "label_set_weights_%d" % wid, [nhidden, nhidden],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "label_set_bias_%d" % wid, [nhidden], initializer=tf.zeros_initializer())
        task_weights['label_set_weights_%d' % wid] = output_weights
        task_weights['label_set_bias_%d' % wid] = output_bias
        learning_rates['label/set/layer%d' % wid] = []
        for sid in range(FLAGS.SGD_K):
            with tf.variable_scope('sgd%d' % sid):
                lr = tf.get_variable("label/set/layer%d/learning_rate" % wid, dtype=tf.float32, shape=(),
                                     initializer=tf.constant_initializer(train_lr),
                                     trainable=FLAGS.is_meta_sgd,
                                     constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
                learning_rates['label/set/layer%d' % wid].append(lr)
                tf.summary.scalar('label_set_layer%d_sgd%d' % (wid, sid), lr)

    if not FLAGS.use_pooled_output:
        tf.logging.info('Creating Per-layer Weights')
        layer_weights = tf.get_variable(
            "bert_layer_weights", [bert_config.num_hidden_layers],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        if not FLAGS.do_eval:
            layer_weights = tf.reshape(tf.nn.softmax(layer_weights), [1, 1, -1])
        proj_weight = tf.get_variable(
            "cls_output_projection", [bert_config.hidden_size, bert_config.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        task_weights['bert_proj_weight'] = proj_weight
        task_weights['bert_layer_weights'] = layer_weights
        learning_rates['bert_layer_weights'] = []
        learning_rates['bert_proj_weight'] = []
        for sid in range(FLAGS.SGD_K):
            with tf.variable_scope('sgd%d' % sid):
                lr = tf.get_variable("bert_layer_weights", dtype=tf.float32, shape=(),
                                     initializer=tf.constant_initializer(train_lr),
                                     trainable=FLAGS.is_meta_sgd,
                                     constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
                learning_rates['bert_layer_weights'].append(lr)
                lr = tf.get_variable("bert_proj_weight", dtype=tf.float32, shape=(),
                                     initializer=tf.constant_initializer(train_lr),
                                     trainable=FLAGS.is_meta_sgd,
                                     constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
                learning_rates['bert_proj_weight'].append(lr)

    if FLAGS.update_only_label_embedding:
        learning_rates['label_embs'] = []
        for sid in range(FLAGS.SGD_K):
            with tf.variable_scope('sgd%d' % sid):
                 lr = tf.get_variable("label_embs/learning_rate", dtype=tf.float32, shape=(),
                                      initializer=tf.constant_initializer(train_lr),
                                      trainable=FLAGS.is_meta_sgd,
                                      constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
                 learning_rates['label_embs'].append(lr)
                 tf.summary.scalar('label_embs_sgd%d' % (sid), lr)

    bert_learning_rates = {}
    for lid in range(bert_config.num_hidden_layers):
        bert_learning_rates[lid] = []
        with tf.variable_scope('layer_%d' % lid):
            for sk in range(FLAGS.SGD_K):
                with tf.variable_scope('sgd%d' % sk):
                    lr = tf.get_variable("learning_rate", dtype=tf.float32, shape=(),
                                         initializer=tf.constant_initializer(train_lr),
                                         trainable=FLAGS.is_meta_sgd,
                                         constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
                    bert_learning_rates[lid].append(lr)
                    tf.summary.scalar('bert_layer%d_sgd%d' % (lid, sk), lr)
    bert_learning_rates["word_embedding"] = []
    bert_learning_rates["pooler"] = []
    for sk in range(FLAGS.SGD_K):
        with tf.variable_scope('sgd%d' % sk):
            lr = tf.get_variable("word_embedding_lr", dtype=tf.float32, shape=(),
                                 initializer=tf.constant_initializer(train_lr),
                                 trainable=FLAGS.is_meta_sgd,
                                 constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
            bert_learning_rates["word_embedding"].append(lr)
            tf.summary.scalar('bert_embeddings_sgd%d' % sk, lr)
            lr = tf.get_variable("pooler_lr", dtype=tf.float32, shape=(),
                                 initializer=tf.constant_initializer(train_lr),
                                 trainable=FLAGS.is_meta_sgd,
                                 constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if FLAGS.clip_lr else None)
            bert_learning_rates["pooler"].append(lr)
            tf.summary.scalar('bert_pooler_sgd%d' % sk, lr)

    return task_weights, learning_rates, bert_learning_rates


def determine_learning_rate(bert_learning_rates, task_learning_rates,key, sgd_id):
    layer_num = re.findall(r'layer_([0-9]+)', key)
    layer_num = int(layer_num[0]) if len(layer_num) != 0 else None
    #import ipdb; ipdb.set_trace();
    if layer_num is not None:
        if FLAGS.do_eval:
            lr = tf.gather(bert_learning_rates[layer_num], sgd_id)
        else:
            lr = bert_learning_rates[layer_num][sgd_id]
    elif len(re.findall('embeddings', key)) > 0:
        if FLAGS.do_eval:
            lr = tf.gather(bert_learning_rates["word_embedding"],sgd_id)
        else:
            lr = bert_learning_rates["word_embedding"][sgd_id]
    elif len(re.findall('pooler', key)) > 0:
        if FLAGS.do_eval:
            lr =  tf.gather(bert_learning_rates["pooler"], sgd_id)
        else:
            lr = bert_learning_rates["pooler"][sgd_id]

    elif len(re.findall('bert_proj_weight', key)) > 0:
        if FLAGS.do_eval:
            lr = tf.gather(task_learning_rates['bert_proj_weight'], sgd_id)
        else:
            lr = task_learning_rates["bert_proj_weight"][sgd_id]
    elif len(re.findall('bert_layer_weights', key)) > 0:
        if FLAGS.do_eval:
            lr = tf.gather(task_learning_rates['bert_layer_weights'], sgd_id)
        else:
            lr = task_learning_rates["bert_layer_weights"][sgd_id]
    elif key == 'label_embs':
        if FLAGS.do_eval:
            lr = tf.gather(task_learning_rates[key], sgd_id)
        else:
            lr = task_learning_rates[key][sgd_id]
    else:
        vals = key.split('_')
        if len(vals) >= 3:
            if len(vals) == 3:
                name, _, num = vals
                tname = name + '/layer' + num
            else:
                name1, name2, _, num = vals
                tname = name1 + '/' + name2 + '/layer' + num
            if tname in task_learning_rates:
                if FLAGS.do_eval:
                    lr = tf.gather(task_learning_rates[tname], sgd_id)
                else:
                    lr = task_learning_rates[tname][sgd_id]
            else:
                tf.logging.info('Using default learning rate for %s' % key)
                if FLAGS.use_exp_lr:
                    lr = np.log(FLAGS.train_lr)
                else:
                    lr = FLAGS.train_lr
        else:
            tf.logging.info('Using default learning rate for %s' % key)
            if FLAGS.use_exp_lr:
                lr = np.log(FLAGS.train_lr)
            else:
                lr = FLAGS.train_lr
    if FLAGS.use_exp_lr:
        lr = tf.exp(lr)
    return lr


def create_maml_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                      labels, use_one_hot_embeddings, tasks_num_labels):
    """
       input_ids: (tasks, batchsize, max_seq_len)
       input_mask: (tasks, batchsize, max_seq_len)
       segment_ids: (tasks, batchsize, max_seq_len)
       labels: (tasks, batchsize)
       tasks_num_labels: (tasks)
    """

    if FLAGS.max_train_batch_size > 0:
        idx = tf.random_shuffle(tf.range(FLAGS.max_train_batch_size))
    else:
        idx = tf.random_shuffle(tf.range(FLAGS.train_batch_size))

    input_ids = tf.gather(input_ids, idx, axis=1)
    # import ipdb; ipdb.set_trace()
    input_mask = tf.gather(input_mask, idx, axis=1)
    segment_ids = tf.gather(segment_ids, idx, axis=1)
    # print_op = tf.print("Before shuffle", {"idx": idx, "labels_before": labels})
    # with tf.control_dependencies([print_op]):
    labels = tf.gather(labels, idx, axis=1)
    # print_op = tf.print("After shuffle", {"idx": idx, "labels_after": labels})
    # with tf.control_dependencies([print_op]):
    tasks_num_labels = tf.gather(tasks_num_labels, idx, axis=1)

    def split_in_half(x):
        # TODO Train batch size should be equal to eval batch size or there should be a eval meta_batchsz
        support, query = x[:, : FLAGS.meta_batchsz], x[:, FLAGS.meta_batchsz:]
        # if FLAGS.shuffle_domain:
        #     query = tf.gather(query, idx)
        return support, query

    # import ipdb; ipdb.set_trace()
    support_ids_x, query_ids_x = split_in_half(input_ids)
    support_mask_x, query_mask_x = split_in_half(input_mask)
    support_segment_ids_x, query_segment_ids_x = split_in_half(segment_ids)
    support_y, query_y = split_in_half(labels)
    tasks_num_labels = tasks_num_labels[:, 0]

    task_weights, task_learning_rates, bert_learning_rates = create_task_weights_and_lr(bert_config,
                                                                                        FLAGS.label_emb_size)

    bert_model = modeling.BertModel(
        config=bert_config,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert")

    def get_learning_rate(key, sgd_id):
        return determine_learning_rate(bert_learning_rates, task_learning_rates, key, sgd_id)

    def meta_task(input, task_idx=None):
        """
        map_fn only support one parameters, so we need to unpack from tuple.
        """
        support_ids, support_mask, support_segment_ids, support_y, \
            query_ids, query_mask, query_segment_ids, query_y, num_labels = input

        # support_ids = tf.Print(support_ids, [support_ids], 'meta_task: support_ids')

        support = (support_ids, support_mask, support_segment_ids, support_y)
        if FLAGS.num_batches > 0:
            # import ipdb; ipdb.set_trace();
            # split support into num_batches
            support_ids = tf.reshape(support_ids, [FLAGS.num_batches,
                                                   support_ids.get_shape().as_list()[0] // FLAGS.num_batches] +
                                     support_ids.get_shape().as_list()[1:])
            support_mask = tf.reshape(support_mask, [FLAGS.num_batches,
                                                     support_mask.get_shape().as_list()[0] // FLAGS.num_batches] +
                                      support_mask.get_shape().as_list()[1:])
            support_segment_ids = tf.reshape(support_segment_ids,
                                             [FLAGS.num_batches,
                                              support_segment_ids.get_shape().as_list()[0] // FLAGS.num_batches] +
                                             support_segment_ids.get_shape().as_list()[1:])
            support_y = tf.reshape(support_y,
                                   [FLAGS.num_batches, support_y.get_shape().as_list()[0] // FLAGS.num_batches] +
                                   support_y.get_shape().as_list()[1:])
            support = (support_ids[0], support_mask[0], support_segment_ids[0], support_y[0])

        query = (query_ids, query_mask, query_segment_ids, query_y)

        support_loss, support_logits, support_acc, label_embs, label_mask = forward(support, bert_model,
                                                                                    is_training, num_labels,
                                                                                    weights=bert_model.weights,
                                                                                    task_weights=task_weights,
                                                                                    reuse=False)
        if FLAGS.update_only_label_embedding:
            current_task_weights = {key: value for key, value in task_weights.items() if "label" not in key}
        else:
            current_task_weights = {key: value for key, value in task_weights.items()}
        # compute gradients
        weights_for_grad = filter_by_layer(bert_model.weights)

        if FLAGS.SGD_K > 0 and FLAGS.sgd_first_batch:
            # import ipdb; ipdb.set_trace()
            grads = tf.gradients(support_loss, list(weights_for_grad.values()))

            task_grads = tf.gradients(support_loss, list(current_task_weights.values()))
            all_grads = grads + task_grads
            if FLAGS.update_only_label_embedding:
                label_emb_grads = tf.gradients(support_loss, label_embs)
                # label_emb_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in label_emb_grads]
                all_grads += label_emb_grads

            if FLAGS.stop_grad:
                new_grads = []
                for g in all_grads:
                    if g is None:
                        new_grads.append(g)
                    elif isinstance(g, tf.IndexedSlices):
                        new_grads.append(
                            tf.IndexedSlices(tf.stop_gradient(g.values), g.indices, dense_shape=g.dense_shape))
                    else:
                        new_grads.append(tf.stop_gradient(g))
                all_grads = new_grads
            else:
                tf.logging.info('Using second order')

            all_grads, _ = tf.clip_by_global_norm(all_grads, clip_norm=1.0)

            grads = all_grads[:len(grads)]
            task_grads = all_grads[len(grads):len(grads) + len(task_grads)]
            label_emb_grads = None
            if FLAGS.update_only_label_embedding:
                label_emb_grads = all_grads[len(grads) + len(task_grads):]

            # grad and variable dict
            gvs = dict(zip(weights_for_grad.keys(), grads))
            task_gvs = dict(zip(current_task_weights.keys(), task_grads))

            # theta = theta - alpha * grads
            # import ipdb; ipdb.set_trace()
            fast_weights = dict(zip(weights_for_grad.keys(),
                                    [weights_for_grad[key] - get_learning_rate(key, 0) * convert_to_tensor(gvs[key])
                                     if gvs[key] is not None else weights_for_grad[key]
                                     for key in weights_for_grad.keys()]))

            fast_task_weights = {key: param - get_learning_rate(key, 0) * convert_to_tensor(task_gvs[key])
                                 for key, param in current_task_weights.items()}
            if FLAGS.update_only_label_embedding:
                label_embs = label_embs - get_learning_rate('label_embs', 0) * label_emb_grads[0]
        else:
            # SGD_K == 0 : no adaptation
            fast_weights = weights_for_grad
            fast_task_weights = current_task_weights
        fast_weights_forward = {key: fast_weights[key] if key in fast_weights else bert_model.weights[key]
                                    for key in bert_model.weights}

        query_loss = tf.constant(0., dtype=tf.float32)
        nquery_steps = 0
        query_weight_sum = 0.
        total_inner_steps = FLAGS.inner_epochs * (FLAGS.num_batches - 1) + 1
        tf.logging.info('total_inner_steps: %d' % total_inner_steps)
        if FLAGS.average_query_loss or FLAGS.warp_layers or FLAGS.prototypical_baseline:
            step_query_loss, query_logits, query_acc, _, _ = forward(query, bert_model, is_training, num_labels,
                                                                     weights=fast_weights_forward,
                                                                     task_weights=fast_task_weights,
                                                                     label_embs=label_embs,
                                                                     label_mask=label_mask,
                                                                     reuse=True)
            # tf.summary.scalar('step%d_query_loss' % 0, step_query_loss)
            tf.logging.info('Query loss at step %d' % 0)
            if FLAGS.weight_query_loss:
                q_wt = (1. / 2**total_inner_steps)
                step_query_loss = q_wt * step_query_loss
                query_weight_sum += q_wt
                # tf.summary.scalar('step%d_query_loss_weight' % 0, q_wt)
                tf.logging.info('\t query loss weight: %.5f' % q_wt)
            query_loss += step_query_loss
            nquery_steps += 1

        lr_multiplier = tf.constant(1.0, dtype=tf.float32)
        num_inner_steps = FLAGS.num_batches - 1
        if FLAGS.randomize_inner_steps and is_training:
            num_inner_steps = tf.random_uniform(shape=(), minval=FLAGS.min_inner_steps,
                                                maxval=FLAGS.num_batches, dtype=tf.int32)

        if is_training:
            inner_epochs = FLAGS.inner_epochs
        else:
            # fixing eval epochs to 5
            inner_epochs = 5

        # continue to build G steps graph
        for ep_id in range(inner_epochs):
            if FLAGS.num_batches > 0 and ep_id > 0:
                # shuffle batches before each epoch
                _idx = tf.random_shuffle(tf.range(FLAGS.num_batches))
                # import ipdb; ipdb.set_trace()
                support_ids = tf.gather(support_ids, _idx, axis=0)
                support_mask = tf.gather(support_mask, _idx, axis=0)
                support_segment_ids = tf.gather(support_segment_ids, _idx, axis=0)
                support_y = tf.gather(support_y, _idx, axis=0)
                # num_labels = tf.gather(num_labels, idx, axis=1)
            for batch_id in range(1, FLAGS.num_batches if FLAGS.num_batches > 0 else FLAGS.SGD_K):
                # we need meta-train loss to fine-tune the task and meta-test loss to update theta
                if FLAGS.num_batches > 0:
                    support = (support_ids[batch_id], support_mask[batch_id], support_segment_ids[batch_id],
                               support_y[batch_id])
                if FLAGS.update_only_label_embedding:
                    # TODO: CORRECT LABEL MASK TO INCLUDE LABEL NOT SEEN DURING FIRST FORWARD PASS
                    support_loss, support_logits, support_acc, label_embs, label_mask = forward(
                        support, bert_model,is_training, num_labels, weights=fast_weights_forward,
                        label_embs=label_embs, label_mask=label_mask, task_weights=fast_task_weights, reuse=True)
                else:
                    support_loss, support_logits, support_acc, label_embs, label_mask = forward(
                        support, bert_model, is_training, num_labels, weights=fast_weights_forward,
                        task_weights=fast_task_weights, reuse=True)
                # compute gradients
                grads = tf.gradients(support_loss, list(fast_weights.values()))
                task_grads = tf.gradients(support_loss, list(fast_task_weights.values()))
                all_grads = grads + task_grads
                if FLAGS.update_only_label_embedding:
                    label_emb_grads = tf.gradients(support_loss, label_embs)
                    # label_emb_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in label_emb_grads]
                    all_grads += label_emb_grads

                if FLAGS.stop_grad:
                    new_grads = []
                    for g in all_grads:
                        if g is None:
                            new_grads.append(g)
                        elif isinstance(g, tf.IndexedSlices):
                            new_grads.append(
                                tf.IndexedSlices(tf.stop_gradient(g.values), g.indices, dense_shape=g.dense_shape))
                        else:
                            new_grads.append(tf.stop_gradient(g))
                    all_grads = new_grads
                else:
                    tf.logging.info('Using second order')

                all_grads, _ = tf.clip_by_global_norm(all_grads, clip_norm=1.0)

                grads = all_grads[:len(grads)]
                task_grads = all_grads[len(grads):len(grads) + len(task_grads)]
                label_emb_grads = None
                if FLAGS.update_only_label_embedding:
                    label_emb_grads = all_grads[len(grads) + len(task_grads):]

                # compose grad and variable dict
                gvs = dict(zip(fast_weights.keys(), grads))
                task_gvs = dict(zip(fast_task_weights.keys(), task_grads))
                # import ipdb; ipdb.set_trace()
                # update theta_pi according to varibles
                if FLAGS.randomize_inner_steps and is_training:
                    lr_multiplier = tf.cond(num_inner_steps < batch_id, lambda: tf.constant(0.), lambda: tf.constant(1.0))
                # print_op = tf.print("Inner step %d" % batch_id, {'lr_multiplier': lr_multiplier,
                #                                                  'num_inner_steps': num_inner_steps})
                lr_id = batch_id if FLAGS.num_batches == FLAGS.SGD_K else 0
                # with tf.control_dependencies([print_op]):
                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - lr_multiplier * get_learning_rate(key, lr_id) *
                                         convert_to_tensor(gvs[key]) if gvs[key] is not None else fast_weights[key]
                                         for key in fast_weights.keys()]))
                # import ipdb; ipdb.set_trace()
                if FLAGS.update_only_label_embedding:
                    not_labels = ["label" not in key for key in fast_task_weights.keys()]
                    assert np.all(not_labels), "label key exists!"

                fast_task_weights = {key: param - lr_multiplier * get_learning_rate(key, lr_id) *
                                     convert_to_tensor(task_gvs[key])
                                     for key, param in current_task_weights.items()}
                # forward on theta_pi
                if FLAGS.update_only_label_embedding:
                    label_embs = label_embs - \
                                 lr_multiplier * get_learning_rate('label_embs', lr_id) * label_emb_grads[0]
                fast_weights_forward = {key: fast_weights[key] if key in fast_weights else bert_model.weights[key]
                                        for key in bert_model.weights}
                # we need accumulate all meta-test losses to update theta
                if (ep_id == FLAGS.inner_epochs - 1 and batch_id == FLAGS.num_batches - 1) or \
                        ((FLAGS.average_query_loss or FLAGS.warp_layers) and
                         (FLAGS.num_batches * ep_id + batch_id) % FLAGS.average_query_every == 0):
                    step_query_loss, query_logits, query_acc, _, _ = forward(query, bert_model, is_training, num_labels,
                                                                             weights=fast_weights_forward,
                                                                             task_weights=fast_task_weights,
                                                                             label_embs=label_embs,
                                                                             label_mask=label_mask,
                                                                             reuse=True)
                    # tf.summary.scalar('step%d_query_loss' % (ep_id * (FLAGS.num_batches - 1) + batch_id),
                    #                   step_query_loss)
                    tf.logging.info('Query loss at step ep_id %d, batch_id %d' % (ep_id, batch_id))
                    if FLAGS.weight_query_loss:
                        q_wt = 1. / 2 ** (total_inner_steps - ep_id * (FLAGS.num_batches - 1) - batch_id)
                        step_query_loss = q_wt * step_query_loss
                        query_weight_sum += q_wt
                        # tf.summary.scalar('step%d_query_loss_weight' % (ep_id * (FLAGS.num_batches - 1) + batch_id),
                        #                   q_wt)
                        tf.logging.info('\t query loss weight: %.5f' % q_wt)
                    query_loss += step_query_loss
                    nquery_steps += 1
                # query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
                # query_preds.append(query_pred)
                # query_losses.append(query_loss)

        if FLAGS.average_query_loss or FLAGS.warp_layers:
            if FLAGS.weight_query_loss:
                query_loss /= query_weight_sum
            else:
                query_loss /= nquery_steps

        query_pred = tf.argmax(query_logits, axis=1, output_type=tf.int64)
        result = [support_loss, support_acc, query_loss, query_acc, query_pred]

        return result

    # meta_task_fn = lambda inps: meta_task(inps, bert_model)
    # task_idxs = tf.constant(np.array([idx for idx in range(FLAGS.tasks_per_gpu)]), dtype=tf.int32)

    if FLAGS.debug:
        support_loss_tasks, support_acc_tasks, query_loss_tasks, query_acc_tasks, query_pred_tasks = [], [], [], [], []
        for task_idx in range(FLAGS.tasks_per_gpu):
            result = meta_task([support_ids_x[task_idx], support_mask_x[task_idx], support_segment_ids_x[task_idx],
                                support_y[task_idx],
                                query_ids_x[task_idx], query_mask_x[task_idx], query_segment_ids_x[task_idx],
                                query_y[task_idx], tasks_num_labels[task_idx]], task_idx=task_idx)
            support_loss_task, support_acc_task, query_loss_task, query_acc_task, query_pred_task = result
            support_loss_tasks.append(support_loss_task)
            support_acc_tasks.append(support_acc_task)
            query_loss_tasks.append(query_loss_task)
            query_acc_tasks.append(query_acc_task)
            query_pred_tasks.append(query_pred_task)

        support_loss_tasks = tf.stack(support_loss_tasks)
        support_acc_tasks = tf.stack(support_acc_tasks)
        query_loss_tasks = tf.stack(query_loss_tasks)
        query_acc_tasks = tf.stack(query_acc_tasks)
        query_pred_tasks = tf.stack(query_pred_tasks)
    else:
        out_dtype = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        result = tf.map_fn(meta_task,
                           elems=(support_ids_x, support_mask_x, support_segment_ids_x, support_y,
                                  query_ids_x, query_mask_x, query_segment_ids_x, query_y, tasks_num_labels),
                           dtype=out_dtype, parallel_iterations=FLAGS.tasks_per_gpu, name='map_fn')

        support_loss_tasks, support_acc_tasks, query_loss_tasks, query_acc_tasks, query_pred_tasks = result
    # import ipdb; ipdb.set_trace()

    if is_training:
        # average loss
        support_loss = tf.reduce_sum(support_loss_tasks) / FLAGS.tasks_per_gpu
        query_loss = tf.reduce_sum(
            query_loss_tasks) / FLAGS.tasks_per_gpu  # [tf.reduce_sum(query_losses_tasks[j]) / FLAGS.tasks_per_gpu
        support_acc = tf.reduce_sum(support_acc_tasks) / FLAGS.tasks_per_gpu
        query_acc = tf.reduce_sum(query_acc_tasks) / FLAGS.tasks_per_gpu

    else:  # eval
        # average loss
        support_loss = tf.reduce_sum(support_loss_tasks) / FLAGS.tasks_per_gpu
        query_loss = tf.reduce_sum(query_loss_tasks) / FLAGS.tasks_per_gpu
        support_acc = tf.reduce_sum(support_acc_tasks) / FLAGS.tasks_per_gpu
        query_acc = tf.reduce_sum(query_acc_tasks) / FLAGS.tasks_per_gpu
        # query_acc = query_acc_tasks
    mode = "Train" if is_training else "Eval"
    tf.summary.scalar(mode + '_support_loss', support_loss)
    tf.summary.scalar(mode + '_support_acc', support_acc)
    tf.summary.scalar(mode + '_query_loss', query_loss)
    tf.summary.scalar(mode + '_query_acc_step', query_acc)

    return query_loss, query_loss_tasks, query_acc, query_pred_tasks


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        num_labels = features["num_labels"]

        # is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        if is_training:
            (total_loss, per_task_loss, query_acc, query_pred) = create_maml_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                use_one_hot_embeddings, num_labels)
        else:
            (total_loss, per_task_loss, query_acc, query_pred) = create_maml_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                use_one_hot_embeddings, num_labels)

        # import ipdb; ipdb.set_trace()

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        init_hook = None
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)
        elif FLAGS.warm_start_model_dir:
            class InitHook(tf.train.SessionRunHook):
                """initializes model from a checkpoint_path
                args:
                    modelPath: full path to checkpoint
                """

                def __init__(self, checkpoint_dir, is_eval=False):
                    self.modelPath = saved_model_utils.get_variables_path(checkpoint_dir)
                    self.is_eval = is_eval
                    self.initialized = False

                def begin(self):
                    """
                    Restore encoder parameters if a pre-trained encoder model is available and we haven't trained previously
                    """
                    if not self.initialized:
                        # checkpoint = tf.train.latest_checkpoint(self.modelPath)
                        checkpoint = self.modelPath
                        if checkpoint is None:
                            tf.logging.info('No pre-trained model is available, training from scratch.')
                        else:
                            tf.logging.info(
                                'Pre-trained model {0} found in {1} - warmstarting.'.format(checkpoint, self.modelPath))
                            tf.train.warm_start(checkpoint)
                        self.initialized = True

            init_hook = InitHook(FLAGS.warm_start_model_dir)


        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, fix_bert=FLAGS.fix_bert)

            training_hooks = []
            if FLAGS.debug:
                logging_hook = tf.train.LoggingTensorHook({"query_loss": total_loss,
                                                           "query_pred": query_pred,
                                                           "label_ids": label_ids,
                                                           },
                                                          every_n_iter=1)
                training_hooks.append(logging_hook)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn,
                training_chief_hooks=[init_hook] if init_hook else None,
                training_hooks=training_hooks)
        elif mode == tf.estimator.ModeKeys.EVAL:

            eval_metrics = {"mean_loss": tf.metrics.mean(total_loss),
                            "eval_accuracy": tf.metrics.mean(query_acc),
                            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": query_pred},
                scaffold=scaffold_fn)
        return output_spec

    return model_fn


def finetune_model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                               num_train_steps, num_warmup_steps, use_tpu,
                               use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s , type = %s" % (name, features[name].shape, features[name].dtype))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        num_labels = features["num_labels"]

        # is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        step = tf.train.get_or_create_global_step()
        train_op = logits = None
        if is_training:
            (total_loss, per_example_loss, train_op) = create_finetune_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                use_one_hot_embeddings, num_labels, num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps
            )
            global_step_incr_op = tf.assign(step, step + 1)
            train_op = tf.group(train_op, global_step_incr_op)
        else:
            total_loss, logits, support_acc, label_embs = create_finetune_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                use_one_hot_embeddings, num_labels,
                num_warmup_steps=num_warmup_steps)
        # import ipdb; ipdb.set_trace()

        tvars = tf.trainable_variables()
        tvars_except_labelemb = [v for v in tvars if "label_embedding" not in v.name]
        # initialized_variable_names = {}
        scaffold_fn = None

        output_spec = None

        class InitHook(tf.train.SessionRunHook):
            """initializes model from a checkpoint_path
            args:
                modelPath: full path to checkpoint
            """

            def __init__(self, checkpoint_dir, is_eval=False):
                self.modelPath = saved_model_utils.get_variables_path(checkpoint_dir)
                self.is_eval = is_eval
                self.initialized = False

            def begin(self):
                """
                Restore encoder parameters if a pre-trained encoder model is available and we haven't trained previously
                """
                if not self.initialized:
                    # checkpoint = tf.train.latest_checkpoint(self.modelPath)
                    checkpoint = self.modelPath
                    if checkpoint is None:
                        tf.logging.info('No pre-trained model is available, training from scratch.')
                    else:
                        tf.logging.info(
                            'Pre-trained model {0} found in {1} - warmstarting.'.format(checkpoint, self.modelPath))
                        tf.train.warm_start(checkpoint, vars_to_warm_start=tvars if self.is_eval else tvars_except_labelemb)
                    self.initialized = True

        if mode == tf.estimator.ModeKeys.TRAIN:
            init_hook = InitHook(FLAGS.warm_start_model_dir)
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss,
                                                       # "per_example_loss": per_example_loss,
                                                       # "label_ids" : label_ids
                                                       },
                                                      every_n_iter=20)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn,
                training_chief_hooks=[init_hook],
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(support_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)
                # import ipdb; ipdb.set_trace()
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions)
                loss = tf.metrics.mean(values=support_loss)
                precision = tf.metrics.precision(labels = label_ids, predictions = predictions)
                recall = tf.metrics.recall(labels = label_ids, predictions=predictions)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "precision" : precision,
                    "recall" : recall
                }

            tvars = tf.trainable_variables()
            # initialized_variable_names = {}
            scaffold_fn = None
            init_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            eval_metrics = metric_fn(total_loss, label_ids, logits)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                # training_chief_hooks=[init_hook],
                scaffold=scaffold_fn)
        else:
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     predictions={"probabilities": probabilities},
            #     scaffold_fn=scaffold_fn)
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)
                # import ipdb; ipdb.set_trace()
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            # eval_metrics = metric_fn(per_example_loss, label_ids, logits,
            #                          is_real_example)
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"prediction": predictions, "actual" : label_ids, "input_ids" : input_ids, "logits" : logits},
                #eval_metric_ops = eval_metrics,
                scaffold=scaffold_fn)

        return output_spec

    return model_fn


def read_data_sizes_from_tfrecord(input_files):
    c = []
    for i, fn in enumerate(input_files):
        nc = 0
        # import ipdb; ipdb.set_trace()
        for record in tf.python_io.tf_record_iterator(fn):
            nc += 1
        c.append(nc)
    total_datapoints = sum(c)
    return total_datapoints, c


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    override_config = {}
    if FLAGS.attention_probs_dropout_prob is not None:
        override_config["attention_probs_dropout_prob"] = FLAGS.attention_probs_dropout_prob
    if FLAGS.hidden_dropout_prob is not None:
        override_config["hidden_dropout_prob"] = FLAGS.hidden_dropout_prob
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file, override_config=override_config)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)


    # task_eval_files = [os.path.join(FLAGS.data_dir, x) for x in FLAGS.task_eval_files.strip().split(",")]
    task_eval_files = [os.path.join(FLAGS.data_dir, x) if x.startswith('amazon')
                       else os.path.join(FLAGS.supervised_data_dir, x)
                       for x in FLAGS.task_eval_files.strip().split(",")]

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        strategy = None
    else:
        tf.logging.info("GPU available: %s" % tf.test.is_gpu_available())
        strategy = tf.contrib.distribute.MirroredStrategy()

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # tpu_config=tf.contrib.tpu.TPUConfig(
    #     iterations_per_loop=FLAGS.iterations_per_loop,
    #     num_shards=FLAGS.num_tpu_cores,
    #     per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    run_config = None
    if FLAGS.do_train:
        # task_train_files = [os.path.join(FLAGS.data_dir, x) for x in FLAGS.task_train_files.strip().split(",")]
        task_train_files = [os.path.join(FLAGS.data_dir, x) if x.startswith('meta_pretrain')
                            else os.path.join(FLAGS.supervised_data_dir, x)
                            for x in FLAGS.task_train_files.strip().split(",")]

        ntrain_examples, nexamples_per_file_train = read_data_sizes_from_tfrecord(task_train_files)
        tf.logging.info('Examples per file:\n' + "\n".join(
            map(lambda x: "%s: %d" % (x[0], x[1]), zip(task_train_files, nexamples_per_file_train))))
        num_train_steps = int(
            ntrain_examples / (FLAGS.train_batch_size * FLAGS.num_gpus) * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        run_config = tf.estimator.RunConfig(
            train_distribute=strategy,
            eval_distribute=None,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=1,
            save_summary_steps=20
        )

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=2,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        # estimator = tf.contrib.tpu.TPUEstimator(
        #     use_tpu=FLAGS.use_tpu,
        #     model_fn=model_fn,
        #     config=run_config,
        #     train_batch_size=FLAGS.train_batch_size,
        #     eval_batch_size=FLAGS.eval_batch_size,
        #     predict_batch_size=FLAGS.predict_batch_size)
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            warm_start_from=FLAGS.warm_start_model_dir,
            params={'train_batch_size': FLAGS.train_batch_size,
                    'eval_batch_size': FLAGS.eval_batch_size})
        
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", ntrain_examples)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = input_fn_builder(
            input_files=task_train_files,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            nexamples_per_file=nexamples_per_file_train
        )
        tf.logging.info("COMPLETED TRAIN_INPUT_FN!!!")

        eval_input_fn = input_fn_builder(
            input_files=task_eval_files,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True
        )
        
        eval_features = {
            "input_ids": tf.placeholder(tf.int64, shape=(None, FLAGS.eval_batch_size, FLAGS.max_seq_length)),
            "input_mask": tf.placeholder(tf.int64, shape=(None, FLAGS.eval_batch_size, FLAGS.max_seq_length)),
            "segment_ids": tf.placeholder(tf.int64, shape=(None, FLAGS.eval_batch_size, FLAGS.max_seq_length)),
            "label_ids": tf.placeholder(tf.int64, shape=(None, FLAGS.eval_batch_size)),
            "num_labels": tf.placeholder(tf.int32, shape=(None, FLAGS.eval_batch_size))
        }

        def compare_eval_fn(best_eval_result, current_eval_result):
            return best_eval_result["eval_accuracy"] <= current_eval_result["eval_accuracy"]

        exporter = tf.estimator.BestExporter(
                name="best_exporter",
                serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(
                    eval_features,
                    default_batch_size=None),
                compare_fn=compare_eval_fn,
                exports_to_keep=1)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=exporter, throttle_secs=2,
                                          steps=None)

        result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval or FLAGS.do_predict:

        neval_examples, nexamples_per_file_eval_train = read_data_sizes_from_tfrecord([task_eval_files[0]])
        neval_steps = (neval_examples / FLAGS.eval_batch_size) * FLAGS.num_train_epochs
        num_warmup_steps = int(FLAGS.warmup_proportion * neval_steps)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", neval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        tf.logging.info("  Num steps = %d", neval_steps)
        run_config = tf.estimator.RunConfig(
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps= neval_steps * FLAGS.num_train_epochs,
            keep_checkpoint_max=1
        )
        #save_checkpoints_steps=1)
        model_fn = finetune_model_fn_builder(
            bert_config=bert_config,
            num_labels=2,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=neval_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

        eval_train_input_fn = input_fn_builder(
            input_files=task_eval_files,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        eval_eval_input_fn = input_fn_builder(
            input_files=task_eval_files,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # import ipdb; ipdb.set_trace()
        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        # estimator = tf.contrib.tpu.TPUEstimator(
        #     use_tpu=FLAGS.use_tpu,
        #     model_fn=model_fn,
        #     config=run_config,
        #     train_batch_size=FLAGS.train_batch_size,
        #     eval_batch_size=FLAGS.eval_batch_size,
        #     predict_batch_size=FLAGS.predict_batch_size)
        estimator = None
        if FLAGS.do_eval:
            estimator = tf.estimator.Estimator(
                model_fn=model_fn,
                config=run_config,
                params={'train_batch_size': FLAGS.train_batch_size,
                        'eval_batch_size': FLAGS.eval_batch_size},
                # warm_start_from=warm_start
                # warm_start_from=FLAGS.warm_start_model_dir
            )
        else:
            estimator = tf.estimator.Estimator(model_fn,
                                               warm_start_from=FLAGS.warm_start_model_dir,
                                               params={'eval_batch_size': FLAGS.eval_batch_size})
        estimator.train(input_fn = eval_train_input_fn)

        estimator.evaluate(input_fn=eval_eval_input_fn)

        import glob
        for fname in glob.glob("%s/model*" % FLAGS.output_dir):
            try:
                os.remove(fname)
            except:
                pass


if __name__ == "__main__":
    np.random.seed(42)
    tf.set_random_seed(42)
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_train_files")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
