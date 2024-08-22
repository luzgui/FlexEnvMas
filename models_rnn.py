# @OldAPIStack
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from typing import Dict, List, Tuple
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util.debug import log_once

from gymnasium.spaces import Dict

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class LSTMActionMaskModel(RecurrentNetwork):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        hiddens_size=256,
        cell_size=64,
    ):
        
        
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )
        
        
        super(LSTMActionMaskModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.cell_size = cell_size

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, orig_space["observations"].shape[0]), name="inputs"
        )
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1"
        )(input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=dense1,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(lstm_out)
        
        values = tf.keras.layers.Dense(1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c],
        )
        # self.rnn_model.summary()


    @override(ModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        # Creating a __init__ function that acts as a passthrough and adding the warning
        # there led to errors probably due to the multiple inheritance. We encountered
        # the same error if we add the Deprecated decorator. We therefore add the
        # deprecation warning here.
        if log_once("recurrent_network_tf"):
            deprecation_warning(
                old="ray.rllib.models.tf.recurrent_net.RecurrentNetwork"
            )
        assert seq_lens is not None
        
        
        action_mask = input_dict["obs"]["action_mask"]
        # flat_inputs = input_dict["obs_flat"]
        flat_inputs = input_dict["obs"]["observations"]
        
        
        
        inputs = add_time_dimension(
            padded_inputs=flat_inputs, seq_lens=seq_lens, framework="tf"
        )
        output, new_state = self.forward_rnn(
            inputs,
            state,
            seq_lens,
        )
        
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        # masked_output = output + inf_mask
        
        masked_output=tf.add(output,tf.reshape(inf_mask, output.shape))
        
        return tf.reshape(masked_output, [-1, self.num_outputs]), new_state
        # return tf.reshape(output, [-1, self.num_outputs]), new_state


    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]


    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])