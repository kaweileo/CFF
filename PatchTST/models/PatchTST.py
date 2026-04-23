__all__ = ['PatchTST']

# Cell
from typing import Optional

from torch import Tensor
from torch import nn

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    """
    初始化模型

    该构造函数用于初始化模型参数和结构。它接受一系列参数，包括配置、序列长度、维度等，
    并根据这些参数设置模型的属性和结构。如果配置了分解，则创建趋势和残差的处理模块；
    否则，创建单一的模型处理输入。

    参数:
    - configs: 包含模型配置的类，如输入维度、序列长度、预测长度等
    - max_seq_len: 最大序列长度，默认为1024
    - d_k, d_v: 注意力机制中的维度，如果未提供则根据d_model计算
    - norm: 归一化方法，默认为'BatchNorm'
    - attn_dropout: 注意力机制中的dropout率，默认为0
    - act: 激活函数，默认为'gelu'
    - key_padding_mask: 是否使用padding mask，默认为'auto'
    - padding_var: padding的变量，默认为None
    - attn_mask: 注意力机制中的mask，默认为None
    - res_attention: 是否使用残差注意力机制，默认为True
    - pre_norm: 是否在注意力机制前进行归一化，默认为False
    - store_attn: 是否存储注意力权重，默认为False
    - pe: 位置编码方法，默认为'zeros'
    - learn_pe: 位置编码是否可学习，默认为True
    - pretrain_head: 是否预训练头部，默认为False
    - head_type: 头部类型，默认为'flatten'
    - verbose: 是否打印详细信息，默认为False
    - **kwargs: 其他参数
    """

    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                                 patch_len=patch_len, stride=stride,
                                                 max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                 n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=attn_dropout,
                                                 dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                 padding_var=padding_var,
                                                 attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                 store_attn=store_attn,
                                                 pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                 head_dropout=head_dropout, padding_patch=padding_patch,
                                                 pretrain_head=pretrain_head, head_type=head_type,
                                                 individual=individual, revin=revin, affine=affine,
                                                 subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                               patch_len=patch_len, stride=stride,
                                               max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                               n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                               attn_dropout=attn_dropout,
                                               dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                               padding_var=padding_var,
                                               attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                               store_attn=store_attn,
                                               pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                               head_dropout=head_dropout, padding_patch=padding_patch,
                                               pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                               revin=revin, affine=affine,
                                               subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           patch_len=patch_len, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x):
    # def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """
        前向传播函数，处理输入数据x以生成模型输出。

        如果启用了分解模式，则将输入数据分解为趋势和残差部分，分别通过各自的模型进行处理；
        否则，直接将输入数据通过模型进行处理。

        参数:
        x: 输入数据，形状为 [Batch, Input length, Channel]，其中 Batch 为批次大小，Input length 为输入数据长度，Channel 为通道数。

        返回:
        形状为 [Batch, Input length, Channel] 的张量，表示模型的输出。
        """
        if self.decomposition:
            # 如果启用了分解模式，则对输入数据进行分解
            res_init, trend_init = self.decomp_module(x)
            # res_init, trend_init = self.decomp_module(batch_x)
            # 将分解后的数据维度调整为 [Batch, Channel, Input length]
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            # 分别对残差和趋势部分进行处理
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            # res = self.model_res(res_init, batch_x_mark, dec_inp, batch_y_mark)
            # trend = self.model_trend(trend_init, batch_x_mark, dec_inp, batch_y_mark)
            # 将处理后的残差和趋势部分相加
            x = res + trend
            # 调整维度回到 [Batch, Input length, Channel]
            x = x.permute(0, 2, 1)
        else:
            # 如果未启用分解模式，直接调整输入数据维度并进行处理
            x = x.permute(0, 2, 1)
            x = self.model(x)
            # x = batch_x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            # x = self.model(x, batch_x_mark, dec_inp, batch_y_mark)
            # 调整维度回到 [Batch, Input length, Channel]
            x = x.permute(0, 2, 1)
        return x