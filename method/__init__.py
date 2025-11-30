from .DEMO import DEMO
from .PredFormer import PredFormer
from .LaST import LaST

from utils import printf, Color as Co

__all__ = ['setup_model', 'methods_dict']

methods_dict = {
    'DEMO': DEMO,
    'PredFormer': PredFormer,
    'LaST': LaST,
}

def setup_model(args):
    """
    This function is used to prepare the model.
    这个函数用于准备模型。
    """
    if args.model_ckpt != "":
        return methods_dict[args.model].load_from_checkpoint(
            checkpoint_path=args.model_ckpt,
            d_model=int(128), s_patch=4, n_heads=8, n_layer=12,
            dropout=0, drop_path=0, s_local=[3, 3, 3],
            s_data_x=args.data_back, s_data_y=args.data_pred, lr=args.lr, r_hidden=args.r_hidden
        )
    model = args.model
    printf(s="Setup model", m=f"{Co.B}Reading model: {Co.Y}{model}{Co.RE}")
    if model is None:
        printf(s="Setup model", err="--model cannot be None", m="please config --model.")
        raise ValueError('Model cannot be None, please config parser.model.')
    elif model == 'DEMO':
        return methods_dict[model](d_model=args.d_model, s_patch=args.s_patch, n_coder=args.n_coder,
                                   n_layer=args.n_layer, s_kernel=args.s_kernel,
                                   s_data_x=args.data_back, s_data_y=args.data_pred, lr=args.lr)
    elif model == 'LaST':
        return methods_dict[model](
            d_model=int(args.d_model), s_patch=args.s_patch, n_heads=args.n_heads, n_layer=args.n_layer,
            s_kernel=args.s_kernel, r_forward=args.r_forward, dropout=args.dropout, drop_path=args.drop_path,
            s_local=args.s_local, data_back=args.data_back, data_pred=args.data_pred, lr=args.lr,
            lr_scheduler=args.lr_scheduler, attn_bias=args.attn_bias,
        )
    elif model == 'PredFormer':
        return methods_dict[model](
            d_model=int(args.d_model), s_patch=args.s_patch, n_heads=args.n_heads, n_layer=args.n_layer,
            r_forward=args.r_forward, dropout=args.dropout, drop_path=args.drop_path,
            data_back=args.data_back, data_pred=args.data_pred, lr=args.lr,
            lr_scheduler=args.lr_scheduler, attn_bias=args.attn_bias,
        )
    else:
        printf(s="Setup model", err=f"--model setup error", m=f"--model {Co.W}{model}{Co.RE} is not supported.")
        raise ValueError(
            f'The given model is not supported, config --model or setup_model() in method/__init__.py.')
    



