import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import os
from scipy.optimize import curve_fit
import inspect

def read_sys_json(path):
    """
    从指定路径读取系统参数 JSON 文件
    参数:
    path (str): 包含 JSON 文件的目录路径。
    返回:
    dict: 包含系统参数数据的 dict
    """
    import json
    file_path = 'sys.json'
    try:
        # 以只读模式打开 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取 JSON 文件内容并转换为字典
            data_dict = json.load(file)
            for key,value in data_dict.items():
                if type(value)==float:
                    print(key,': {:,}'.format(value))
                else:
                    print(key,':',value)
        return data_dict
    except FileNotFoundError:
        print(f"未找到文件: {file_path}")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的 JSON 格式。")

def read_id900_binary(path,channel):
    """
    从指定路径读取特定通道的二进制文件，并将其转换为 Pandas DataFrame。

    参数:
    path (str): 包含二进制文件的目录路径。
    channel (int): 要读取的通道号。

    返回:
    pandas.DataFrame: 包含二进制文件数据的 DataFrame,列名为 'time_satmp' 和 'start_index'。
    None: 如果未找到指定通道的二进制文件。
    """
    
    for file_name in os.listdir(path):
        if 'C{}'.format(channel) in file_name:
            pmt_array=np.fromfile('{}/{}'.format(path,file_name),dtype=np.uint64).reshape(-1,2)
            pmt_pd=pd.DataFrame(pmt_array,columns=['time_satmp','start_index'])
            return pmt_pd
    print('No channel {} bin file found'.format(channel))
    return None

def window_index(df, start, end, pulse_num=1, pulse_interval=5e6):
    """
    生成一个布尔索引，用于筛选出指定数据框中时间戳在一系列时间窗口内的行。

    参数:
    df (pandas.DataFrame): 输入的数据框，包含 'time_satmp' 和 'start_index' 列。
    start (int or float): 第一个时间窗口的起始时间。
    end (int or float): 第一个时间窗口的结束时间。
    pulse_num (int, 可选): 脉冲的数量，默认为 1。
    pulse_interval (float, 可选): 相邻时间窗口之间的间隔[ps]，默认为 5us。

    返回:
    pandas.Series: 一个布尔索引，用于筛选出符合条件的行。
    """
    # 初始化一个布尔索引，所有元素初始化为 False
    _index = df['start_index'] < 0
    # 循环生成多个时间窗口的布尔索引
    for pulse_i in range(pulse_num):
        # 计算当前时间窗口的起始和结束时间
        _window_start = start + pulse_i * pulse_interval
        _window_end = end + pulse_i * pulse_interval
        # 生成当前时间窗口的布尔索引，并累加到总索引上
        _index += (df['time_satmp'] >= _window_start) & (df['time_satmp'] <= _window_end)
    return _index


def count2gamma(df):
    """
    该函数用于将 PMT 计数数据转换为 gamma 值、gamma 误差和相位值。

    参数:
    df (pandas.DataFrame): 包含 PMT 计数数据的 DataFrame，应包含 'pmt1'、'pmt2' 和 'turn' 列。

    返回:
    pandas.DataFrame: 包含计算后的 'gamma'、'gamma_err' 和 'phase' 列的 DataFrame。
    """
    # 计算 gamma 值，公式为 (pmt1 - pmt2) / (pmt1 + pmt2)
    df['gamma']=(df['pmt1']-df['pmt2'])/(df['pmt1']+df['pmt2'])
    # 计算 gamma 误差的平方，公式为 (2 * pmt2 / ((pmt1 + pmt2) ** 2) * sqrt(pmt1)) ** 2 + (-2 * pmt1 / ((pmt1 + pmt2) ** 2) * sqrt(pmt2)) ** 2
    df['gamma_err']=(2*df['pmt2']/((df['pmt1']+df['pmt2'])**2)*np.sqrt(df['pmt1']))**2+(-2*df['pmt1']/((df['pmt1']+df['pmt2'])**2)*np.sqrt(df['pmt2']))**2
    # 对 gamma 误差的平方取平方根，得到 gamma 误差
    df['gamma_err']=np.sqrt(df['gamma_err'])
    # 计算相位值，公式为 turn * 360 * 2
    df['phase']=df['turn']*360*2



# 定义模型函数：正弦波
def model_func_count_offset(x, gamma, phase_offset,count_offset):
    """
    A sin(x + phi) + B
    """
    scale=1
    return gamma*(np.sin(scale*x/180*np.pi + phase_offset))+count_offset

def model_func_sin(x, gamma, phase_offset):
    """
    A sin(x + phi)
    """
    scale=1
    return gamma*(np.sin(scale*x/180*np.pi + phase_offset))


# 生成数据点
def estimate_gamma(df,model_func=model_func_count_offset):
    """  A cos(df + B)
    return estimate [A,B]"""
    x_data = df['turn']*360*2  # x轴数据
    y_data = df['gamma']  # y轴数据
    y_err = df['gamma_err']  # y轴数据误差
    pmt1 = df['pmt1']
    pmt2 = df['pmt2']
    x_data_arange = np.linspace(0, 360*2, 200)  # 生成更多数据点用于拟合
    # 使用curve_fit进行非线性最小二乘法拟合
    # 定义参数的下界和上界
    # 获取函数的参数信息
    sig = inspect.signature(model_func)
    # 获取参数的数量
    param_count = len(sig.parameters)
    if param_count == 2+1:
        lower_bounds = [0, 0]  # 参数 a 和 b 的下界
        upper_bounds = [1, 2*np.pi]  # 参数 a 和 b 的上界
        params, covariance = curve_fit(model_func, x_data, y_data,bounds=(lower_bounds, upper_bounds),sigma=y_err)
    elif param_count == 3+1:
        lower_bounds = [0, 0,-0.1]  # 参数 a 和 b 的下界
        upper_bounds = [1, 2*np.pi,0.1]  # 参数 a 和 b 的上界
        params, covariance = curve_fit(model_func, x_data, y_data,bounds=(lower_bounds, upper_bounds),sigma=y_err)
    else:
        print('model_func has {} parameters, but only 2 or 3 parameters are supported'.format(param_count))
        params, covariance = curve_fit(model_func, x_data, y_data,sigma=y_err)

    # 使用拟合参数计算预测值
    y_pred = model_func(x_data, *params)

    # 计算R²
    mean_y = np.mean(y_data)
    total_variance = np.sum((y_data - mean_y) ** 2)
    residual_variance = np.sum((y_data - y_pred) ** 2)
    r_squared = 1 - (residual_variance / total_variance)
    # 输出拟合参数
    if param_count == 2+1:
        print(f"Fitted parameters: gamma={params[0]:.3f}±{np.sqrt(np.diag(covariance))[0]:.3f}, phase_offset={params[1]:.3f}±{np.sqrt(np.diag(covariance))[1]:.3f}, R²={r_squared:.3%}")
    elif param_count == 3+1:
        print(f"Fitted parameters: gamma={params[0]:.3f}±{np.sqrt(np.diag(covariance))[0]:.3f}, phase_offset={params[1]:.3f}±{np.sqrt(np.diag(covariance))[1]:.3f}, count_offset={params[1]:.3f}±{np.sqrt(np.diag(covariance))[1]:.3f}, R²={r_squared:.3%}")

    # 绘制原始数据点和拟合曲线
    plt.subplot(211)
    plt.errorbar(x_data,pmt1,fmt='-o',yerr=np.sqrt(pmt1),label='pmt1',capsize=5)
    plt.errorbar(x_data,pmt2,fmt='-o',yerr=np.sqrt(pmt2),label='pmt2',capsize=5)
    plt.plot(x_data,(pmt1+pmt2)/2, 'x-.', label='(pmt1+pmt2)/2')
    plt.xlim(0,360*2)
    plt.xlabel('phase offset with 729 OAM turn [°]')
    plt.ylabel('pmt counts')
    plt.title('PMT count, experimental error estimate by shot noise')
    plt.legend(loc=1)
    plt.grid()
    plt.subplot(212)
    plt.errorbar(x_data, (y_data), fmt='o', label='experimental data',yerr=y_err,capsize=5)
    plt.plot(x_data_arange,(model_func(x_data_arange, *params)), '-', label='fit {:.1%}±{:.1%}'.format(np.abs(params[0]),np.sqrt(np.diag(covariance))[0]))
    plt.xlabel('phase offset with 729 OAM turn [°] {:+.2f}°'.format(params[1]/np.pi*180))
    plt.ylabel('estimae gamma')
    plt.title('LS estimae $\\gamma$: {:.3%}±{:.3%}, $R^2=${:.1%}'.format(np.abs(params[0]),np.sqrt(np.diag(covariance))[0],r_squared))
    plt.xlim(0,360*2)
    plt.legend(loc=1)
    plt.grid()
    plt.tight_layout()
    params_err=np.sqrt(np.diag(covariance))

    return params,params_err