class MHA(nn.Module):

    def __init__(self, dmodel, dq, dk, dv, heads):
        super(MHA, self).__init__()
        self.heads = heads
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.dropout = nn.Dropout(p = 0.1)

        self.WQ = nn.Parameter(self.init_param(dmodel, heads * dq, seed = 43))
        self.WK = nn.Parameter(self.init_param(dmodel, heads * dk, seed = 44))
        self.WV = nn.Parameter(self.init_param(dmodel, heads * dv, seed = 45))
        self.WO = nn.Parameter(self.init_param(heads * dv, dmodel, seed = 46))

        self.scaling_factor = math.sqrt(dk)

    def init_param(self, *size, seed):
        torch.manual_seed(seed)
        return torch.randn(size)

    def forward(self, H = None):
        BS, T, _ = H.shape

        Q = torch.matmul(H, self.WQ.T)
        K = torch.matmul(H, self.WK.T)
        V = torch.matmul(H, self.WV.T)

        Q = Q.view(BS, T, self.heads, self.dq).transpose(1, 2)
        K = K.view(BS, T, self.heads, self.dk).transpose(1, 2)
        V = V.view(BS, T, self.heads, self.dv).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling_factor
        attention = F.softmax(attention_scores, dim = -1)
        # attention = self.dropout(attention)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(BS, T, -1)
        out = torch.matmul(out, self.WO.T)
        return out


class FFN(nn.Module):

    def __init__(self, dmodel, d_ff):
        super(FFN, self).__init__()
        torch.manual_seed(47)
        self.linear1 = nn.Linear(dmodel, d_ff)
        self.dropout = nn.Dropout(p = 0.1)
        torch.manual_seed(48)
        self.linear2 = nn.Linear(d_ff, dmodel)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.linear1.weight, mode = 'fan_out', nonlinearity = 'relu')
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight, mode = 'fan_out', nonlinearity = 'relu')
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        out = self.linear2(x)
        return out


class OutputLayer(nn.Module):

    def __init__(self, dmodel, vocab_size):
        super(OutputLayer, self).__init__()
        torch.manual_seed(49)
        self.linear = nn.Linear(dmodel, vocab_size)
        self.dropout = nn.Dropout(p = 0.1)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, representations):
        out = self.linear(representations)
        # out = self.dropout(out)
        return out


class MHCA(nn.Module):

    def __init__(self, dmodel, dq, dk, dv, heads):
        super(MHCA, self).__init__()
        self.heads = heads
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.dropout = nn.Dropout(p = 0.1)

        self.WQ = nn.Parameter(self.init_param(dmodel, heads * dq, seed = 43))
        self.WK = nn.Parameter(self.init_param(dmodel, heads * dk, seed = 44))
        self.WV = nn.Parameter(self.init_param(dmodel, heads * dv, seed = 45))
        self.WO = nn.Parameter(self.init_param(heads * dv, dmodel, seed = 46))
        self.scaling_factor = math.sqrt(dk)

    def init_param(self, *size, seed):
        torch.manual_seed(seed)
        return torch.randn(size)

    def forward(self, decoder_input, encoder_output):
        BS, T_dec, _ = decoder_input.shape
        _, T_enc, _ = encoder_output.shape

        Q = torch.matmul(decoder_input, self.WQ.T).view(BS, T_dec, self.heads, -1).transpose(1, 2)
        K = torch.matmul(encoder_output, self.WK.T).view(BS, T_enc, self.heads, -1).transpose(1, 2)
        V = torch.matmul(encoder_output, self.WV.T).view(BS, T_enc, self.heads, -1).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling_factor
        attention = F.softmax(attention_scores, dim = -1)
        # attention = self.dropout(attention)

        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(BS, T_dec, -1)
        out = torch.matmul(out, self.WO.T)
        return out


class MHMA(nn.Module):

    def __init__(self, dmodel, dq, dk, dv, heads, mask = None):
        super(MHMA, self).__init__()
        self.heads = heads
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.mask = mask
        self.dropout = nn.Dropout(p = 0.1)

        self.WQ = nn.Parameter(self.init_param(heads * dq, dmodel, seed = 43))
        self.WK = nn.Parameter(self.init_param(heads * dk, dmodel, seed = 44))
        self.WV = nn.Parameter(self.init_param(heads * dv, dmodel, seed = 45))
        self.WO = nn.Parameter(self.init_param(dmodel, heads * dv, seed = 46))
        self.scaling_factor = math.sqrt(dk)

    def init_param(self, *size, seed):
        torch.manual_seed(seed)
        return torch.randn(size)

    def forward(self, x, mask=None):
        BS, T, _ = x.shape
        Q = torch.matmul(x, self.WQ.T).view(BS, T, self.heads, -1).transpose(1, 2)
        K = torch.matmul(x, self.WK.T).view(BS, T, self.heads, -1).transpose(1, 2)
        V = torch.matmul(x, self.WV.T).view(BS, T, self.heads, -1).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling_factor

        if mask is None:
            mask = torch.triu(torch.ones((T, T), device = x.device, dtype = torch.bool), diagonal = 1)
            mask = mask[None, None, :, :].expand(BS, self.heads, T, T)
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        else :
            attention_scores = attention_scores + mask

        attention = F.softmax(attention_scores, dim=-1)
        # attention = self.dropout(attention)

        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(BS, T, -1)
        out = torch.matmul(out, self.WO.T)
        return out
