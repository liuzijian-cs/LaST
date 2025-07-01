import re


class Color:
    K = '\033[30m'  # Black(黑色)
    R = '\033[31m'  # Red(红色): Error
    G = '\033[32m'  # Green(绿色):Function
    B = '\033[34m'  # Blue(蓝色) System Messages
    Y = '\033[33m'  # Yellow(黄色): File name
    P = '\033[35m'  # Purple(紫色): State
    C = '\033[36m'  # Cyan(青色): Keys
    W = '\033[37m'  # White(白色):
    RE = '\033[0m'  # Reset(重置)


Co = Color  # Rename


def printf(s: str = None, m: str = None, f: str = "console_output.txt", log: bool = True, err: str = ''):
    """
    Utility function to output information to the console and save it to a log file if specified.
    功能函数，用于将信息输出至控制台并根据需要保存到日志文件中。

    :param s: The state or context of the message. (消息的状态或上下文)
    :param m: The message to be displayed. (要显示的信息)
    :param f: The file path to save the log if logging is enabled. (如果启用了日志记录，则日志文件的路径)
    :param log: A boolean flag to indicate whether to log the message to a file. (布尔标志，指示是否将消息记录到文件中)
    :param err: An error prefix to append before the main message if an error occurs. (可选的错误前缀，将其添加到主要信息前)
    """
    # s(state), m(message),f(file), log(log to file)
    message = f"{Co.P}[{s:<18}]{Co.RE} : {m}" if err == '' else f"{Co.P}[{s:<18}]{Co.RE} : {Co.R}{err} >>> {m}{Co.RE}"
    if log and f is not None:
        with open(f, 'a') as f:
            f.write(re.sub(r'\x1b\[[0-9;]*m', '', message+'\n'))
            f.flush()
    print(message)
