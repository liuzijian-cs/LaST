import os
import re
import yaml
import datetime
import argparse
import multiprocessing
from utils import Color as Co
from utils import printf
from method import setup_model
from method import methods_dict
from data import data_dict


# Function List:
# | - Co: Color
# | - printf: Utility function to output information to the console and save it to a log file if specified.
#             功能函数，用于将信息输出至控制台并根据需要保存到日志文件中。

def auto_allocation_worker(args):
    if args.num_workers != 0:
        return args
    if os.name == 'nt':  # Windows file system
        args.num_workers = 0
        printf(
            s="setup args",
            m=f"{Co.B}Auto-configure num_worker(when num_worker=0), System[{Co.C}Windows{Co.B}], num_worker keep to {Co.C}0{Co.RE}"
        )
    elif os.name == 'posix':
        args.num_workers = multiprocessing.cpu_count()
        if args.ddp and args.n_device > 1:
            args.num_workers = args.num_workers // args.n_device
            if args.num_workers > 32:
                args.num_workers = 32
        printf(
            s="setup args",
            m=f"{Co.B}Auto-configure num_worker(when num_worker=0), System[{Co.C}Linux{Co.B}], num_worker is changed to {Co.C}{args.num_workers}{Co.RE}"
        )
    else:
        args.num_workers = args.num_workers
    return args


def setup_working_dir(args, version=None):
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Check that the working path to be used exists(检查即将使用的工作路径存在)
    working_dir = args.working_path if args.working_path is not None else 'working_path'
    project_dir = args.project_name if args.project_name is not None else 'project_default'
    working_path = os.path.join(working_dir, project_dir)
    os.makedirs(working_path, exist_ok=True)
    # Version Control(版本控制)
    if version is None:
        args.version = 0
        args.version_name = "default" if args.version_name == "" else args.version_name
        max_version = -1
        pattern = re.compile(r'^v(\d{3})_')
        for dir_name in os.listdir(working_path):
            match = pattern.match(dir_name)
            if match:
                version = int(match.group(1))
                if version > max_version:
                    max_version = version
        args.version = max_version + 1
        args.version = f"v{args.version:03d}"
        version_dir = args.version + "_" + args.version_name + "_" + current_time
        args.version_path = os.path.abspath(os.path.join(working_path, version_dir))
    else:
        args.version = version
        version_dir = str(args.version) + "_" + args.version_name + "_ckpt_" + current_time
        args.version_path = os.path.abspath(os.path.join(working_path, version_dir))


def setup_ckpt(args):
    match = re.search(r'v\d{3}', args.ckpt)
    version = match.group(0) if match else None
    setup_working_dir(args, version=version)
    return args


def setup_args(args, parser):  # todo
    excluded_keys = {'args', 'ckpt', 'version', 'version_path'}  # 设置不应更新的参数键
    try:
        with open(args.args, 'r') as f:
            args_dict = yaml.safe_load(f)
            for key, val in args_dict.items():
                if key not in excluded_keys:
                    # 获取当前 args 的值
                    current_val = getattr(args, key, None)
                    # 比较 YAML 中的值与当前 args 的值
                    if current_val != val:
                        printf(
                            s="setup args",
                            m=f"{Co.G}Updating args {Co.C}'--{key}' {Co.G}from {Co.C}{current_val}{Co.C}{Co.G} to {Co.C}{val}{Co.RE}"
                        )
                    if isinstance(val, bool):
                        if val:
                            parser.set_defaults(**{key: True})
                        else:
                            parser.set_defaults(**{key: False})
                    else:
                        parser.set_defaults(**{key: val})
    except FileNotFoundError:
        printf(s="Setup args", err="FileNotFoundError", m=f"--args file {Co.W}[{args.args}]{Co.RE} is not exist.")
        raise FileNotFoundError
    except yaml.YAMLError as e:
        printf(s="Setup args", err="yaml.YAMLError", m=f"{e}")
        raise e
    args = parser.parse_args()  # Console arguments take precedence (命令行参数优先)
    return args, parser


def setup_conf(args):
    if args.conf in data_dict:
        if args.conf in ['t2m', 'uv10', 'tcc', 'r']:  # todo: 5_625only
            args.conf = os.path.abspath(os.path.join(args.data_root, 'WeatherBench', '5_625', f'{args.conf}_conf.yaml'))
        elif args.conf in ['sevir', 'sevirlr']:
            args.conf = os.path.abspath(os.path.join(args.data_root, 'SEVIR', f'{args.conf}_conf.yaml'))
        else:
            args.conf = os.path.abspath(os.path.join(args.data_root, args.conf, 'conf.yaml'))

    excluded_keys = {'conf', 'version', 'version_path'}  # 设置不应更新的参数键
    try:
        with open(args.conf, 'r') as f:
            conf_dict = yaml.safe_load(f)
            for key, val in conf_dict.items():
                if key in excluded_keys:
                    continue
                if getattr(args, key, None) != val:
                    printf(s="setup conf",
                           m=f"{Co.G}Updating args {Co.C}'--{key}' {Co.G}from {Co.C}{getattr(args, key, None)}{Co.C}{Co.G} to {Co.C}{val}{Co.RE}")
                    setattr(args, key, val)
    except FileNotFoundError:
        printf(s="Setup conf", err="FileNotFoundError",
               m=f"--conf file {Co.W}[{args.conf}]{Co.RE} is not exist, the default arguments in --args will be used.")
    except yaml.YAMLError as e:
        printf(s="Setup conf", err="yaml.YAMLError", m=f"{e}")
        raise e
    return args


def setup_experiment(parser):
    # Initialize the console output.txt file(初始化 console_output.txt 文件)
    with open("console_output.txt", "w") as _:
        pass

    # Parameter parsing
    args = parser.parse_args()

    # Processing --args file:
    if args.args != '':
        printf(s="Setup experiment",
               m=f"{Co.B}Using --args file: {Co.W}{args.args}{Co.B}, note that the console arguments are not changed{Co.RE}")
        args, parser = setup_args(args, parser)

    # Auto Allocation num_workers
    args = auto_allocation_worker(args)

    # Continue training(继续训练)
    if args.ckpt != '':
        printf(s="Setup experiment", m=f"{Co.B}Using --ckpt file: {Co.W}{args.ckpt}{Co.RE}")
        return setup_ckpt(args)

    # Create Working Dir (./working_path/project_name/version_name/)
    setup_working_dir(args, version=None)

    # Processing --conf file:
    if args.conf != '':
        printf(s="Setup experiment", m=f"{Co.B}Using --conf: {Co.Y}{args.conf}{Co.RE}")
        args = setup_conf(args)

    return args
