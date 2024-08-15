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
# - Add extra argument

"""Configuration and hyperparameter sweeps vqgan trainer."""

from maskgit.configs import base_config
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = base_config.get_config()
    config.model_class = "vqgan"
    config.d_step_per_g_step = 1
    config.image_size = 256
    config.batch_size = 256
    config.eval_batch_size = 128
    config.eval_every_steps = 10_000
    config.num_train_steps = 1_000_000
    config.pretrained_image_model = True
    config.perceptual_loss_weight = 0.1
    config.perceptual_loss_on_logit = False
    config.eval_exact_match = True
    config.eval_num = 50_000

    config.optimizer.lr = None
    config.optimizer.beta1 = 0.0
    config.optimizer.beta2 = 0.99
    config.optimizer.g_lr = 0.0001
    config.optimizer.d_lr = 0.0001

    config.vqgan = ml_collections.ConfigDict()
    config.vqgan.loss_type = "non-saturating"
    config.vqgan.g_adversarial_loss_weight = 0.1
    config.vqgan.gradient_penalty = "r1"
    config.vqgan.grad_penalty_cost = 10.0

    config.vqvae = ml_collections.ConfigDict()
    config.vqvae.quantizer = "vq"
    config.vqvae.codebook_size = 1024

    config.vqvae.entropy_loss_ratio = 0.1
    config.vqvae.entropy_temperature = 0.01
    config.vqvae.entropy_loss_type = "softmax"
    config.vqvae.commitment_cost = 0.25

    config.vqvae.input_dim = 3
    config.vqvae.filters = 128
    config.vqvae.num_res_blocks = 2
    config.vqvae.channel_multipliers = [1, 1, 2, 2, 4]
    config.vqvae.embedding_dim = 256
    config.vqvae.conv_downsample = False
    config.vqvae.activation_fn = "swish"
    config.vqvae.norm_type = "GN"

    config.discriminator = ml_collections.ConfigDict()
    config.discriminator.channel_multiplier = 1
    config.discriminator.blur_resample = True

    config.tau_anneal = ml_collections.ConfigDict()
    config.tau_anneal.tau_max = 1.0
    config.tau_anneal.tau_min = 0.6
    config.tau_anneal.tau_warmup_steps = 0
    config.tau_anneal.tau_decay_steps = 100_000
    return config


def get_hyper(h):
    return h.product([], name="config")
