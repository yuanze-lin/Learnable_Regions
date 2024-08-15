# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base configuration and hyperparameters."""

import ml_collections

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.image_size = 256
    config.batch_size = 256
    config.eval_batch_size = 64
    config.dtype = "float32"
    config.dataset = "imagenet2012"
    config.save_train_iter = False

    config.experiment = ""
    config.polyak_decay = 0.999
    config.shuffle_buffer_size = 1000
    config.train_shuffle = True
    config.eval_pad_last_batch = False
    config.batch_norm_group_size = -1
    config.pretrained_image_model = False

    config.eval_num = 30000
    config.eval_avg_num = 1
    config.eval_exact_match = True
    config.eval_is_splits = 1

    config.num_train_steps = 1_000_000
    config.eval_every_steps = 100
    config.eval_random_show = True
    config.eval_show_num = 16
    config.checkpoint_every_steps = 10_000

    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.type = "adam"
    config.optimizer.lr = 0.0001
    config.optimizer.beta1 = 0.0
    config.optimizer.beta2 = 0.999

    config.optimizer.lr_schedule = "constant"
    config.optimizer.lr_max = 1e-4
    config.optimizer.lr_min = 0.0
    config.optimizer.lr_warmup_steps = 2_000
    config.optimizer.lr_decay_steps = 98_000
    return config


def get_hyper(h):
    return h.product([
        h.sweep("trial", range(1)),
    ], name="config")
