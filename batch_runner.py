import subprocess
import time

# Task list, executed in order (任务列表，按顺序执行)
commands = [
    "python main.py --num_workers 32 --logger_wandb --args args1.yaml",
    "python main.py --num_workers 32 --logger_wandb --args args2.yaml",
]

# Task status management (任务状态管理)
task_status = ["Waiting"] * len(commands)  # Initial status is "Waiting"（初始状态为“Waiting”）


def print_task_status(current_task_idx, status, time_cost=""):
    """
    Print the status of all tasks. (打印当前所有任务的状态)
    """
    print("\n================= 任务状态 =================")
    for idx, cmd in enumerate(commands):
        if idx < current_task_idx:
            print(f"[Task{idx + 1}] \"{cmd}\" [Successful] [time cost={task_status[idx]}s]")
        elif idx == current_task_idx:
            print(f"[Task{idx + 1}] \"{cmd}\" [{status}] [time cost={time_cost}]")
        else:
            print(f"[Task{idx + 1}] \"{cmd}\" [Waiting]")
    print("==========================================\n")


def execute_task(cmd, task_idx):
    """
    Execute a single task, catch exceptions, and record runtime. (执行单个任务，捕获异常，并记录运行时间)
    """
    try:
        print_task_status(task_idx, "Running")
        start_time = time.time()
        # Execute command (执行命令)
        subprocess.run(cmd, shell=True, check=True)
        end_time = time.time()
        task_status[task_idx] = round(end_time - start_time, 2)  # Record execution time（记录执行时间）
        print_task_status(task_idx, "Successful", task_status[task_idx])
    except subprocess.CalledProcessError as e:
        # Catch exception and print error information (捕获异常并打印错误信息)
        print_task_status(task_idx, "Failed")
        print(f"[Error] 任务失败：{cmd}\n错误信息: {str(e)}")
        task_status[task_idx] = "Failed"


def main():
    print("开始按顺序执行任务...\n")

    # # Optional: add a 2-hour delay before the first task（可选：在第一个任务之前添加 2 小时延时）
    # print("等待 2 小时...\n")
    # time.sleep(2 * 60 * 60)  # 2 hours = 2 * 60 minutes * 60 seconds（2 小时 = 2 * 60 分钟 * 60 秒）

    # Execute each task sequentially（依次执行每个任务）
    for idx, command in enumerate(commands):
        execute_task(command, idx)

    print("\n所有任务执行完成！")


if __name__ == "__main__":
    main()
