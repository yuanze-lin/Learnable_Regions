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
#
# Modifications by Henrique Morimitsu
# - Change default values

r"""Configuration and hyperparameter sweeps for maskgit training."""

from maskgit.configs import base_config
from maskgit.configs import vqgan_config
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = base_config.get_config()
    config.experiment = "maskgit_class_cond"
    config.model_class = "maskgit_class_cond"
    config.sequence_order = "horizontal"

    config.num_class = 1000
    config.batch_size = 256
    config.eval_batch_size = 1
    config.eval_every_steps = 10_000
    config.checkpoint_every_steps = 10_000
    config.eval_show_num = 16
    config.image_size = 256
    config.num_train_steps = 2_000_000
    config.eval_num = 50_000
    config.eval_avg_num = 3
    config.eval_exact_match = True

    config.compute_loss_for_all = False
    config.label_smoothing = 0.1
    config.mask_scheduling_method = "cosine"
    config.sample_num_iterations = 12
    #config.sample_choice_temperature = 4.5
    config.sample_choice_temperature = 0.01
    config.min_masking_rate = 0.5

    config.optimizer.lr = 0.0001
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.96
    config.optimizer.warmup_steps = 5000
    config.optimizer.weight_decay = 4.5e-2

    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_layers = 24
    config.transformer.patch_size = 16
    config.transformer.num_embeds = 768
    config.transformer.intermediate_size = 3072
    config.transformer.num_heads = 16
    config.transformer.dropout_rate = 0.1
    config.transformer.mask_token_id = 2024
    config.transformer.latent_size = 16

    vqgan_cf = vqgan_config.get_config()
    config.vqgan = vqgan_cf.vqgan
    config.vqvae = vqgan_cf.vqvae

    return config


def get_hyper(h):
    return h.product([
        h.sweep("image_size", [256, 512]),
        h.sweep("compute_loss_for_all", [True, False]),
    ], name="config")
