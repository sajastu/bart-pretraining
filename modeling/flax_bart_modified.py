# import math
# import random
# from functools import partial
# from typing import Callable, Optional, Tuple
#
# import numpy as np
#
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# from flax.linen import combine_masks, make_causal_mask
# from flax.linen.attention import dot_product_attention_weights
# from flax.traverse_util import flatten_dict, unflatten_dict
# from jax import lax
# from jax.random import PRNGKey
#
# from transformers import FlaxBartPreTrainedModel
# from transformers.modeling_flax_outputs import (
#     FlaxBaseModelOutput,
#     FlaxBaseModelOutputWithPastAndCrossAttentions,
#     FlaxCausalLMOutputWithCrossAttentions,
#     FlaxSeq2SeqLMOutput,
#     FlaxSeq2SeqModelOutput,
#     FlaxSeq2SeqQuestionAnsweringModelOutput,
#     FlaxSeq2SeqSequenceClassifierOutput,
# )
# from transformers.modeling_flax_utils import (
#     ACT2FN,
#     FlaxPreTrainedModel,
#     append_call_sample_docstring,
#     append_replace_return_docstrings,
#     overwrite_call_docstring,
# )
# from transformers.models.bart.modeling_flax_bart import FlaxBartForConditionalGenerationModule, \
#     BART_DECODE_INPUTS_DOCSTRING, shift_tokens_right, FlaxBartForCausalLMModule, FlaxBartDecoderPreTrainedModel, \
#     FlaxBartDecoderWrapper, FlaxBartModule
# from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# from transformers.models.bart.configuration_bart import BartConfig
#
# class FlaxBartForConditionalGenerationModule(nn.Module):
#     config: BartConfig
#     dtype: jnp.dtype = jnp.float32
#     bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros
#
#     def setup(self):
#
#         self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
#         self.lm_head = nn.Dense(
#             self.model.shared.num_embeddings,
#             use_bias=False,
#             dtype=self.dtype,
#             kernel_init=jax.nn.initializers.normal(self.config.init_std),
#         )
#         self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))
#
#     def _get_encoder_module(self):
#         return self.model.encoder
#
#     def _get_decoder_module(self):
#         return self.model.decoder
#
#     def __call__(
#         self,
#         input_ids,
#         attention_mask,
#         decoder_input_ids,
#         decoder_attention_mask,
#         position_ids,
#         decoder_position_ids,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#         deterministic: bool = True,
#     ):
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             position_ids=position_ids,
#             decoder_position_ids=decoder_position_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             deterministic=deterministic,
#         )
#
#         hidden_states = outputs[0]
#
#         if self.config.tie_word_embeddings:
#             shared_embedding = self.model.variables["params"]["shared"]["embedding"]
#             lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
#         else:
#             lm_logits = self.lm_head(hidden_states)
#
#         lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))
#
#         if not return_dict:
#             output = (lm_logits,) + outputs[1:]
#             return output
#
#         return FlaxSeq2SeqLMOutput(
#             logits=lm_logits,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )
#
#
# class FlaxBartPreTrainedModel(FlaxPreTrainedModel):
#     config_class = BartConfig
#     base_model_prefix: str = "model"
#     module_class: nn.Module = None
#
#     def __init__(
#         self,
#         config: BartConfig,
#         input_shape: Tuple[int] = (1, 1),
#         seed: int = 0,
#         dtype: jnp.dtype = jnp.float32,
#         _do_init: bool = True,
#         **kwargs
#     ):
#         module = self.module_class(config=config, dtype=dtype, **kwargs)
#         super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
#
#     def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
#         # init input tensors
#         input_ids = jnp.zeros(input_shape, dtype="i4")
#         # make sure initialization pass will work for FlaxBartForSequenceClassificationModule
#         input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
#         attention_mask = jnp.ones_like(input_ids)
#         decoder_input_ids = input_ids
#         decoder_attention_mask = jnp.ones_like(input_ids)
#
#         batch_size, sequence_length = input_ids.shape
#         position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
#         decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
#
#         params_rng, dropout_rng = jax.random.split(rng)
#         rngs = {"params": params_rng, "dropout": dropout_rng}
#
#         random_params = self.module.init(
#             rngs,
#             input_ids,
#             attention_mask,
#             decoder_input_ids,
#             decoder_attention_mask,
#             position_ids,
#             decoder_position_ids,
#         )["params"]
#
#         if params is not None:
#             random_params = flatten_dict(unfreeze(random_params))
#             params = flatten_dict(unfreeze(params))
#             for missing_key in self._missing_keys:
#                 params[missing_key] = random_params[missing_key]
#             self._missing_keys = set()
#             return freeze(unflatten_dict(params))
#         else:
#             return random_params
#
#     def init_cache(self, batch_size, max_length, encoder_outputs):
#         r"""
#         Args:
#             batch_size (`int`):
#                 batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
#             max_length (`int`):
#                 maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
#                 cache.
#             encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
#                 `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
#                 `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
#                 is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
#                 cross-attention of the decoder.
#         """
#         # init input variables to retrieve cache
#         decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
#         decoder_attention_mask = jnp.ones_like(decoder_input_ids)
#         decoder_position_ids = jnp.broadcast_to(
#             jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
#         )
#
#         def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
#             decoder_module = module._get_decoder_module()
#             return decoder_module(
#                 decoder_input_ids,
#                 decoder_attention_mask,
#                 decoder_position_ids,
#                 **kwargs,
#             )
#
#         init_variables = self.module.init(
#             jax.random.PRNGKey(0),
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             decoder_position_ids=decoder_position_ids,
#             encoder_hidden_states=encoder_outputs[0],
#             init_cache=True,
#             method=_decoder_forward,  # we only need to call the decoder to init the cache
#         )
#         return unfreeze(init_variables["cache"])
#
#     @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=BartConfig)
#     def encode(
#         self,
#         input_ids: jnp.ndarray,
#         attention_mask: Optional[jnp.ndarray] = None,
#         position_ids: Optional[jnp.ndarray] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         train: bool = False,
#         params: dict = None,
#         dropout_rng: PRNGKey = None,
#     ):
#         r"""
#         Returns:
#
#         Example:
#
#         ```python
#         >>> from transformers import BartTokenizer, FlaxBartForConditionalGeneration
#
#         >>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
#         >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
#
#         >>> text = "My friends are cool but they eat too many carbs."
#         >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
#         >>> encoder_outputs = model.encode(**inputs)
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.return_dict
#
#         if attention_mask is None:
#             attention_mask = jnp.ones_like(input_ids)
#         if position_ids is None:
#             batch_size, sequence_length = input_ids.shape
#             position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
#
#         # Handle any PRNG if needed
#         rngs = {}
#         if dropout_rng is not None:
#             rngs["dropout"] = dropout_rng
#
#         def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
#             encode_module = module._get_encoder_module()
#             return encode_module(input_ids, attention_mask, position_ids, **kwargs)
#
#         return self.module.apply(
#             {"params": params or self.params},
#             input_ids=jnp.array(input_ids, dtype="i4"),
#             attention_mask=jnp.array(attention_mask, dtype="i4"),
#             position_ids=jnp.array(position_ids, dtype="i4"),
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             deterministic=not train,
#             rngs=rngs,
#             method=_encoder_forward,
#         )
#
#     @add_start_docstrings(BART_DECODE_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=BartConfig)
#     def decode(
#         self,
#         decoder_input_ids,
#         encoder_outputs,
#         encoder_attention_mask: Optional[jnp.ndarray] = None,
#         decoder_attention_mask: Optional[jnp.ndarray] = None,
#         decoder_position_ids: Optional[jnp.ndarray] = None,
#         past_key_values: dict = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         train: bool = False,
#         params: dict = None,
#         dropout_rng: PRNGKey = None,
#     ):
#         r"""
#         Returns:
#
#         Example:
#
#         ```python
#         >>> import jax.numpy as jnp
#         >>> from transformers import BartTokenizer, FlaxBartForConditionalGeneration
#
#         >>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
#         >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
#
#         >>> text = "My friends are cool but they eat too many carbs."
#         >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
#         >>> encoder_outputs = model.encode(**inputs)
#
#         >>> decoder_start_token_id = model.config.decoder_start_token_id
#         >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
#
#         >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
#         >>> last_decoder_hidden_states = outputs.last_hidden_state
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.return_dict
#
#         encoder_hidden_states = encoder_outputs[0]
#         if encoder_attention_mask is None:
#             batch_size, sequence_length = encoder_hidden_states.shape[:2]
#             encoder_attention_mask = jnp.ones((batch_size, sequence_length))
#
#         batch_size, sequence_length = decoder_input_ids.shape
#         if decoder_attention_mask is None:
#             decoder_attention_mask = jnp.ones((batch_size, sequence_length))
#
#         if decoder_position_ids is None:
#             if past_key_values is not None:
#                 raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")
#
#             decoder_position_ids = jnp.broadcast_to(
#                 jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
#             )
#
#         # Handle any PRNG if needed
#         rngs = {}
#         if dropout_rng is not None:
#             rngs["dropout"] = dropout_rng
#
#         inputs = {"params": params or self.params}
#
#         # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
#         # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
#         # it can be changed by FlaxBartAttention module
#         if past_key_values:
#             inputs["cache"] = past_key_values
#             mutable = ["cache"]
#         else:
#             mutable = False
#
#         def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
#             decoder_module = module._get_decoder_module()
#             return decoder_module(
#                 decoder_input_ids,
#                 decoder_attention_mask,
#                 decoder_position_ids,
#                 **kwargs,
#             )
#
#         outputs = self.module.apply(
#             inputs,
#             decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
#             decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
#             decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             deterministic=not train,
#             rngs=rngs,
#             mutable=mutable,
#             method=_decoder_forward,
#         )
#
#         # add updated cache to model output
#         if past_key_values is not None and return_dict:
#             outputs, past = outputs
#             outputs["past_key_values"] = unfreeze(past["cache"])
#             return outputs
#         elif past_key_values is not None and not return_dict:
#             outputs, past = outputs
#             outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]
#
#         return outputs
#
#     def __call__(
#         self,
#         input_ids: jnp.ndarray,
#         attention_mask: Optional[jnp.ndarray] = None,
#         decoder_input_ids: Optional[jnp.ndarray] = None,
#         decoder_attention_mask: Optional[jnp.ndarray] = None,
#         position_ids: Optional[jnp.ndarray] = None,
#         decoder_position_ids: Optional[jnp.ndarray] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         train: bool = False,
#         params: dict = None,
#         dropout_rng: PRNGKey = None,
#     ):
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.return_dict
#
#         # prepare encoder inputs
#         if attention_mask is None:
#             attention_mask = jnp.ones_like(input_ids)
#         if position_ids is None:
#             batch_size, sequence_length = input_ids.shape
#             position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
#
#         # prepare decoder inputs
#         if decoder_input_ids is None:
#             decoder_input_ids = shift_tokens_right(
#                 input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
#             )
#         if decoder_attention_mask is None:
#             decoder_attention_mask = jnp.ones_like(decoder_input_ids)
#         if decoder_position_ids is None:
#             batch_size, sequence_length = decoder_input_ids.shape
#             decoder_position_ids = jnp.broadcast_to(
#                 jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
#             )
#
#         # Handle any PRNG if needed
#         rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}
#
#         output = self.module.apply(
#             {"params": params or self.params},
#             input_ids=jnp.array(input_ids, dtype="i4"),
#             attention_mask=jnp.array(attention_mask, dtype="i4"),
#             position_ids=jnp.array(position_ids, dtype="i4"),
#             decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
#             decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
#             decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             deterministic=not train,
#             rngs=rngs,
#         )
#
#         return output
#
#
#
# #######
#
# class MyFlaxBartForConditionalGeneration(FlaxBartPreTrainedModel):
#     module_class = FlaxBartForConditionalGenerationModule
#     dtype: jnp.dtype = jnp.float32
#
#     # def __init__(self, **kwargs):
#     #     import pdb;pdb.set_trace()
#     #     super().__init__(**kwargs)
#
#     @add_start_docstrings(BART_DECODE_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=BartConfig)
#     def decode(
#         self,
#         decoder_input_ids,
#         encoder_outputs,
#         encoder_attention_mask: Optional[jnp.ndarray] = None,
#         decoder_attention_mask: Optional[jnp.ndarray] = None,
#         decoder_position_ids: Optional[jnp.ndarray] = None,
#         past_key_values: dict = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         train: bool = False,
#         params: dict = None,
#         dropout_rng: PRNGKey = None,
#     ):
#         r"""
#         Returns:
#
#         Example:
#
#         ```python
#         >>> import jax.numpy as jnp
#         >>> from transformers import BartTokenizer, FlaxBartForConditionalGeneration
#
#         >>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
#         >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
#
#         >>> text = "My friends are cool but they eat too many carbs."
#         >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
#         >>> encoder_outputs = model.encode(**inputs)
#
#         >>> decoder_start_token_id = model.config.decoder_start_token_id
#         >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
#
#         >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
#         >>> logits = outputs.logits
#         ```"""
#
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.return_dict
#
#         encoder_hidden_states = encoder_outputs[0]
#         if encoder_attention_mask is None:
#             batch_size, sequence_length = encoder_hidden_states.shape[:2]
#             encoder_attention_mask = jnp.ones((batch_size, sequence_length))
#
#         batch_size, sequence_length = decoder_input_ids.shape
#         if decoder_attention_mask is None:
#             decoder_attention_mask = jnp.ones((batch_size, sequence_length))
#
#         if decoder_position_ids is None:
#             if past_key_values is not None:
#                 raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")
#
#             decoder_position_ids = jnp.broadcast_to(
#                 jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
#             )
#
#         # Handle any PRNG if needed
#         rngs = {}
#         if dropout_rng is not None:
#             rngs["dropout"] = dropout_rng
#
#         inputs = {"params": params or self.params}
#
#         # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
#         # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
#         # it can be changed by FlaxBartAttention module
#         if past_key_values:
#             inputs["cache"] = past_key_values
#             mutable = ["cache"]
#         else:
#             mutable = False
#
#         def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
#             decoder_module = module._get_decoder_module()
#             outputs = decoder_module(
#                 decoder_input_ids,
#                 decoder_attention_mask,
#                 decoder_position_ids,
#                 **kwargs,
#             )
#             hidden_states = outputs[0]
#
#             if self.config.tie_word_embeddings:
#                 shared_embedding = module.model.variables["params"]["shared"]["embedding"]
#                 lm_logits = module.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
#             else:
#                 lm_logits = module.lm_head(hidden_states)
#
#             lm_logits += module.final_logits_bias.astype(self.dtype)
#
#             return lm_logits, outputs
#
#         outputs = self.module.apply(
#             inputs,
#             decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
#             decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
#             decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             deterministic=not train,
#             rngs=rngs,
#             mutable=mutable,
#             method=_decoder_forward,
#         )
#
#         if past_key_values is None:
#             lm_logits, decoder_outputs = outputs
#         else:
#             (lm_logits, decoder_outputs), past = outputs
#
#         if return_dict:
#             outputs = FlaxCausalLMOutputWithCrossAttentions(
#                 logits=lm_logits,
#                 hidden_states=decoder_outputs.hidden_states,
#                 attentions=decoder_outputs.attentions,
#                 cross_attentions=decoder_outputs.cross_attentions,
#             )
#         else:
#             outputs = (lm_logits,) + decoder_outputs[1:]
#
#         # add updated cache to model output
#         if past_key_values is not None and return_dict:
#             outputs["past_key_values"] = unfreeze(past["cache"])
#             return outputs
#         elif past_key_values is not None and not return_dict:
#             outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]
#
#         return outputs
#
#     def prepare_inputs_for_generation(
#         self,
#         decoder_input_ids,
#         max_length,
#         attention_mask: Optional[jnp.DeviceArray] = None,
#         decoder_attention_mask: Optional[jnp.DeviceArray] = None,
#         encoder_outputs=None,
#         **kwargs
#     ):
#         # initializing the cache
#         batch_size, seq_length = decoder_input_ids.shape
#
#         past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
#         # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
#         # But since the decoder uses a causal mask, those positions are masked anyways.
#         # Thus we can create a single static attention_mask here, which is more efficient for compilation
#         extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
#         if decoder_attention_mask is not None:
#             position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
#             extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
#         else:
#             position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
#
#         return {
#             "past_key_values": past_key_values,
#             "encoder_outputs": encoder_outputs,
#             "encoder_attention_mask": attention_mask,
#             "decoder_attention_mask": extended_attention_mask,
#             "decoder_position_ids": position_ids,
#         }
#
#     def update_inputs_for_generation(self, model_outputs, model_kwargs):
#         model_kwargs["past_key_values"] = model_outputs.past_key_values
#         model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
#         return model_kwargs
