from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
import numpy as np
from layers import ConvNorm, LinearNorm
from utils import to_device, get_mask_from_lengths


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.hparams = hparams
        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, total_length=None):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        if self.hparams.enable_pack_padded_sequence:
            # pytorch tensor are not reversible, hence the conversion
            input_lengths = input_lengths.cpu().numpy()
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        if self.hparams.enable_pack_padded_sequence:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=total_length)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim + hparams.latent_z_output_dim + hparams.observed_z_output_dim,
            hparams.decoder_rnn_dim, 1)

        self.residual = LinearNorm(hparams.decoder_rnn_dim,
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim + hparams.latent_z_output_dim + hparams.observed_z_output_dim)

        self.linear_projection = LinearNorm(
            # hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            # hparams.decoder_rnn_dim + hparams.attention_rnn_dim + hparams.encoder_embedding_dim + hparams.latent_z_output_dim + hparams.observed_z_output_dim,
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim + hparams.latent_z_output_dim + hparams.observed_z_output_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        if isinstance(alignments, list):
            alignments = torch.stack(alignments)
        alignments.transpose_(0, 1)
        # (T_out, B) -> (B, T_out)
        if isinstance(gate_outputs, list):
            gate_outputs = torch.stack(gate_outputs)

        gate_outputs.transpose_(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        if isinstance(mel_outputs, list):
            mel_outputs = torch.stack(mel_outputs)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, z_latent, z_observed):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context, z_latent, z_observed), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        # decoder_hidden_attention_context = torch.cat(
        #     (self.decoder_hidden,
        #     #  self.attention_context
        #     decoder_input
        #      ), dim=1)

        decoder_hidden_attention_context = F.relu(self.residual(self.decoder_hidden) + decoder_input)

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        decoder_hidden_attention_context_gate = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context_gate)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths, z_latent, z_observed,
        max_memory_length=None, max_output_length=None):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        batch_size = decoder_inputs.size(0)

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=get_mask_from_lengths(memory_lengths, max_len=max_memory_length, invert=True))

        mel_outputs, gate_outputs, alignments = [], [], []
        import torch_xla.debug.metrics as met
        import torch_xla_py.xla_model as xm
        for i in range(max_output_length):
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input, z_latent, z_observed)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
            #print('Step {}'.format(i))
            #print(met.metrics_report())
            #if i % 100 == 0:
            #    xm.mark_step()

        #print(met.metrics_report())
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, z_latent, z_observed):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input, z_latent, z_observed)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

class LatentEncoder(nn.Module):
    def __init__(self, hparams, output_dim):
        super(LatentEncoder, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.latent_embedding_dim,
                         kernel_size=hparams.latent_kernel_size, stride=hparams.latent_stride,
                         padding=int((hparams.latent_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.latent_embedding_dim))
        )
        self.hparams = hparams
        for i in range(1, hparams.latent_n_convolutions):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.latent_embedding_dim,
                             hparams.latent_embedding_dim,
                             kernel_size=hparams.latent_kernel_size,
                             stride=hparams.latent_stride,
                             padding=int((hparams.latent_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.latent_embedding_dim))
            )

        self.gru = nn.GRU(hparams.latent_embedding_dim,
                            int(hparams.latent_rnn_dim / 2), hparams.latent_n_rnns,
                            batch_first=True, bidirectional=True)

        self.mu_linear_projection = LinearNorm(hparams.latent_rnn_dim, output_dim)
        self.logvar_linear_projection = LinearNorm(hparams.latent_rnn_dim, output_dim)

    def forward(self, x, input_lengths, max_length=None):
        # x = x.transpose(1, 2)   # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)   # (B, n_mel_channels, T_out) -> (B, T_out, latent_embedding_dim)

        if not self.hparams.enable_pack_padded_sequence:
            x_sorted = x
        else:
            conv_length = input_lengths
            total_length = max_length
            for i in range(self.hparams.latent_n_convolutions):
                conv_length = torch.div((conv_length + 2 * int((self.hparams.latent_kernel_size - 1) / 2) -
                    (self.hparams.latent_kernel_size - 1) - 1), self.hparams.latent_stride) + 1
                if total_length is not None:
                    total_length = (total_length + 2 * int((self.hparams.latent_kernel_size - 1) / 2) -
                        (self.hparams.latent_kernel_size - 1) - 1) // self.hparams.latent_stride + 1

            device = conv_length.device
            input_lengths_sorted, inds = conv_length.sort(dim=0, descending=True)
            inds = inds.to(device)
            gather_inds = inds.unsqueeze(1).repeat([1, x.size()[1]]).unsqueeze(2).repeat([1, 1, x.size()[2]])
            x_sorted = x.gather(0, gather_inds)

            # pytorch tensor are not reversible, hence the conversion
            input_lengths_sorted_cpu = input_lengths_sorted.cpu().numpy()
            x_sorted = nn.utils.rnn.pack_padded_sequence(
                x_sorted, input_lengths_sorted_cpu, batch_first=True)

        self.gru.flatten_parameters()
        outputs, _ = self.gru(x_sorted)

        if self.hparams.enable_pack_padded_sequence:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=total_length)

        mask = get_mask_from_lengths(input_lengths_sorted).unsqueeze(-1).float()
        outputs = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
        # outputs = outputs.mean(dim=1)
        mu = self.mu_linear_projection(outputs)
        logvar = self.logvar_linear_projection(outputs)

        if not self.hparams.enable_pack_padded_sequence:
            return mu, logvar
        else:
            mu_unsorted, logvar_unsorted = torch.zeros_like(mu), torch.zeros_like(logvar)
            scatter_inds = inds.unsqueeze(1).repeat([1, mu.size()[1]])
            mu_unsorted.scatter_(0, scatter_inds, mu)
            logvar_unsorted.scatter_(0, scatter_inds, logvar)

            return mu_unsorted, logvar_unsorted

    def inference(self, x):
        # x = x.transpose(1, 2)   # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)   # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)

        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        outputs = outputs.mean(dim=1)
        mu, logvar = self.mu_linear_projection(outputs), self.logvar_linear_projection(outputs)
        return mu, logvar


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch, max_input_length=None, max_output_length=None, device=None):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_device(text_padded, device=device, dtype=torch.long)
        input_lengths = to_device(input_lengths, device=device, dtype=torch.long)
        if max_input_length is None:
            max_input_length = torch.max(input_lengths.data).item()
        mel_padded = to_device(mel_padded, device=device, dtype=torch.float)
        gate_padded = to_device(gate_padded, device=device, dtype=torch.float)
        output_lengths = to_device(output_lengths, device=device, dtype=torch.long)
        if max_output_length is None:
            max_output_length = torch.max(output_lengths.data).item()

        return (
                (text_padded, input_lengths, max_input_length, mel_padded, output_lengths, max_output_length),
                (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None, max_length=None):
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths, max_length, invert=True)
            mask = mask.unsqueeze(0).expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs, z_latent, z_observed):
        text_inputs, text_lengths, max_input_length, mels,output_lengths, max_output_length = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths, max_input_length)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths,
            z_latent=z_latent, z_observed=z_observed, max_memory_length=max_input_length,
            max_output_length=max_output_length)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths, max_output_length)

    def inference(self, inputs, z_latent, z_observed):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, z_latent, z_observed)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

class VAE(nn.Module):
    def __init__(self, hparams):
        super(VAE, self).__init__()
        self.hparams = hparams
        y_dim, z_dim = hparams.latent_y_output_dim, hparams.latent_z_output_dim
        self.latent_prior_mu = nn.Parameter(torch.zeros((y_dim, z_dim), requires_grad=True))
        self.latent_prior_sigma = nn.Parameter(torch.ones((y_dim, z_dim), requires_grad=True) * hparams.latent_sigma_init)
        y_dim, z_dim = hparams.observed_y_output_dim, hparams.observed_z_output_dim
        self.observed_prior_mu = nn.Parameter(torch.zeros((y_dim, z_dim), requires_grad=True))
        self.observed_prior_sigma = nn.Parameter(torch.ones((y_dim, z_dim), requires_grad=True) * hparams.observed_sigma_init)

        self.latent_z = LatentEncoder(hparams, hparams.latent_z_output_dim)
        self.observed_z = LatentEncoder(hparams, hparams.observed_z_output_dim)

        self.synthesizer = Tacotron2(hparams)

        torch.nn.init.uniform_(self.latent_prior_mu, a=-0.5, b=0.5)
        torch.nn.init.uniform_(self.observed_prior_mu, a=-0.5, b=0.5)

    def parse_batch(self, batch, max_input_length=None, max_output_length=None, device=None):
        speaker_ids = batch[-1]
        synthesizer_batch = self.synthesizer.parse_batch(batch[:-1], max_input_length,
            max_output_length, device)
        speaker_ids = to_device(speaker_ids, device=device, dtype=torch.long)

        return synthesizer_batch, speaker_ids

    def forward(self, inputs):
        (text_inputs, text_lengths, max_input_length, mels,
            output_lengths, max_output_length) = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

#        latent_z_mu, latent_z_logvar = self.latent_z(mels, output_lengths, max_output_length)

#        observed_z_mu, observed_z_logvar = self.observed_z(mels, output_lengths, max_output_length)

#        z_latent = MultivariateNormal(latent_z_mu, scale_tril=(0.5 * latent_z_logvar).exp().diag_embed()).rsample()
#        z_observed = MultivariateNormal(observed_z_mu, scale_tril=(0.5 * observed_z_logvar).exp().diag_embed()).rsample()
        z_latent = torch.zeros([mels.size(0), self.hparams.latent_z_output_dim]).to(inputs[0].device)
        z_observed = torch.zeros([mels.size(0), self.hparams.observed_z_output_dim]).to(inputs[0].device)
        latent_z_mu, latent_z_logvar = z_latent, z_latent
        observed_z_mu, observed_z_logvar = z_observed, z_observed

        return (self.synthesizer(inputs, z_latent, z_observed),
            (latent_z_mu, latent_z_logvar),
            (observed_z_mu, observed_z_logvar),
            (self.latent_prior_mu, self.latent_prior_sigma),
            (self.observed_prior_mu, self.observed_prior_sigma))

    def inference(self, inputs, z_latent, z_observed):
        return self.synthesizer.inference(inputs, z_latent, z_observed)
