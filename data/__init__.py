from utils import printf, Color as Co
from .MovingMNIST import MovingMNISTDataModule
from .KTH import KTHDataModule
from .TaxiBJ import TaxiBJDataModule
from .WeatherBench import WeatherBenchDataModule
from .SEVIR import SEVIRDataModule
from .Human import HumanDataModule
from .KittiCaltech import KittiCaltechDataModule
from .SeaWaveCora import SeaWaveCoraDataModule

__all__ = ['setup_data', 'data_dict']

data_dict = {
    'MovingMNIST': MovingMNISTDataModule,
    'KTH': KTHDataModule,
    'TaxiBJ': TaxiBJDataModule,
    't2m': WeatherBenchDataModule,
    'uv10': WeatherBenchDataModule,
    'tcc': WeatherBenchDataModule,
    'r': WeatherBenchDataModule,
    'sevir': SEVIRDataModule,
    'sevirlr': SEVIRDataModule,
    'Human': HumanDataModule,
    'KittiCaltech': KittiCaltechDataModule,
    'SeaWaveCora': SeaWaveCoraDataModule,
}


def setup_data(args):
    data = args.data
    printf(s="Setup data", m=f"{Co.B}Reading data: {Co.Y}{data}{Co.RE}")
    basic_config = {
        'data_root': args.data_root,
        'batch_size': args.batch_size,
        'val_batch_size': args.val_batch_size,
        'num_workers': args.num_workers,
        'len_back': args.data_back[0],
        'len_pred': args.data_pred[0],
    }
    if data is None:
        printf(s="Setup data", err='ValueError', m=f'--data {Co.C}[{data}]{Co.RE} cannot be None.')
        raise ValueError('Data cannot be None, please config parser.data.')
    elif data == 'MovingMNIST':
        return data_dict[data](**basic_config, img_size=64, dataset_length=1e4)
    elif data == 'KTH':
        return data_dict[data](**basic_config, img_size=128)
    elif data in ['TaxiBJ', 'Human', 'KittiCaltech', 'SeaWaveCora']:
        return data_dict[data](**basic_config)
    elif data in ['t2m', 'uv10', 'tcc', 'r']:
        return data_dict[data](**basic_config, data=data, data_split='5_625')
    elif data in ['sevir', 'sevirlr']:
        return data_dict[data](**basic_config, data=data)
    else:
        printf(s="Setup data", err='ValueError', m=f'--data {Co.C}[{data}]{Co.RE} Data Module is not exist.')
        raise ValueError('The given data is not supported, config parser.data or data reader in data/__init__.py.')
