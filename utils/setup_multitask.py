import os
import yaml
import glob
import traceback
from .common import printf, Color as Co
from .setup_experiment import setup_args


def print_task_state(args_files, task_state, running_index):
    for i, task in enumerate(task_state):
        if task:
            printf(s='Setup multitask',
                   m=f'{Co.B}|- {Co.W}{args_files[i]:<16}{Co.B}  [{Co.G}Finished{Co.B}]{Co.RE}')
        elif i == running_index:
            printf(s='Setup multitask',
                   m=f'{Co.B}|- {Co.W}{args_files[i]:<16}{Co.B}  [{Co.Y}Running...{Co.B}]{Co.RE}')
        elif i < running_index and not task:
            printf(s='Setup multitask',
                   m=f'{Co.B}|- {Co.W}{args_files[i]:<16}{Co.B}  [{Co.R}Failed{Co.B}]{Co.RE}')
        else:
            printf(s='Setup multitask',
                   m=f'{Co.B}|- {Co.W}{args_files[i]:<16}{Co.B}  [{Co.Y}Waiting...{Co.B}]{Co.RE}')


def setup_multitask(args, parser, run):
    args_files = glob.glob(os.path.join(args.batch_args_path, "*.yaml"))
    printf(s='Setup multitask',
           m=f'{Co.B}Starting multi-task process, {Co.C}{len(args_files)}{Co.B} tasks detected, beginning execution.{Co.RE}')
    printf(s='Setup multitask',
           m=f'{Co.Y}Note that the console does not accept any arguments other than --batch_args_path in multitasking mode, so please configure args properly. {Co.RE}')
    task_state = [False] * len(args_files)
    error_log = []
    for i, args_file in enumerate(args_files):
        try:
            print_task_state(args_files, task_state, i)
            parser.set_defaults(args=args_file)
            # args.args = args_file
            # parser.set_defaults(**{args: args_file})
            run(parser)
            task_state[i] = True
        except Exception as e:
            error_message = f"Error in {args_file}: {str(e)}\n"
            error_log.append(error_message)
            error_log.append(traceback.format_exc())
            print(error_message)
            with open("multitask_error.txt", "a") as f:
                f.writelines(error_message)
                f.flush()
            continue  # Skip to the next YAML file
    printf(s='Setup multitask', m=f'{Co.B}Multi-task process completed.{Co.RE}')
    printf(s='Setup multitask',
           m=f'{Co.B}A total of {len(args_files)} tasks were detected ({Co.G}{sum(task_state)} tasks succeeded{Co.B}, {Co.R}{len(args_files) - sum(task_state)} tasks failed{Co.B}):{Co.RE}')
    print_task_state(args_files, task_state, len(args_files) + 1)
