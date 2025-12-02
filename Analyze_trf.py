# -*- coding: utf-8 -*-
import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import matplotlib
from matplotlib import rcParams





# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置matplotlib参数避免中文问题
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']


def stroop_erp_analysis(file_path, event_dict, output_dir='results'):
    """
    主分析函数：执行Stroop范式的ERP分析
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义分析参数
    tmin = -0.2  # 刺激前200ms
    tmax = 0.8  # 刺激后800ms

    print("=" * 50)
    print("Stroop ERP Analysis Start")
    print("=" * 50)

    # ==========================================
    # 1. 读取数据
    # ==========================================
    print("Step 1: Reading Curry file...")
    print(f"File path: {file_path}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")

        # 显示当前工作目录
        print(f"Current working directory: {os.getcwd()}")

        # 尝试在项目目录中查找
        project_dir = "D:/AllLanProject/python/DataPro1"
        search_pattern = os.path.join(project_dir, "**", "*.cdt")
        cdt_files = glob.glob(search_pattern, recursive=True)

        if cdt_files:
            print("Found .cdt files:")
            for f in cdt_files:
                print(f"  {f}")
            file_path = cdt_files[0]
            print(f"Using file: {file_path}")
        else:
            print("No .cdt files found in project directory")
            return None

    try:
        raw = mne.io.read_raw_curry(file_path, preload=True)
        print(f"Successfully read file: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # 显示基本信息
    print(f"Data info: {len(raw.ch_names)} channels, Sampling rate: {raw.info['sfreq']}Hz")
    print(f"Data duration: {raw.times[-1]:.1f} seconds")
    print(f"Channel examples: {raw.ch_names[:5]}")

    # ==========================================
    # 2. 数据预处理
    # ==========================================
    print("\nStep 2: Data preprocessing...")

    # 带通滤波 (0.1-15Hz)
    raw.filter(l_freq=0.1, h_freq=15.0, fir_design='firwin')
    print("Filtering completed (0.1-15Hz)")

    # 重参考 (平均参考)
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    print("Re-referencing completed (average reference)")

    # ==========================================
    # 3. 事件提取
    # ==========================================
    print("\nStep 3: Extracting event markers...")

    # 获取事件标记(自动）
    events, mapping = mne.events_from_annotations(raw)
    print(f"Found event markers: {mapping}")

    if len(events) == 0:
        print("Warning: No event markers found")
        if raw.annotations:
            print("Raw annotations:", raw.annotations.description)

    # 重新映射事件ID
    real_event_ids = {}
    for label, code in event_dict.items():
        code_str = str(code)
        if code_str in mapping:
            real_event_ids[label] = mapping[code_str]
            print(f"Found marker {code} -> {label} (ID: {mapping[code_str]})")
        else:
            # 尝试反向查找
            found = False
            for key, value in mapping.items():
                if key == str(code) or value == code:
                    real_event_ids[label] = value
                    print(f"Found marker {key} -> {label} (ID: {value})")
                    found = True
                    break
            if not found:
                print(f"Marker {code} ({label}) not found")

    # 如果没有找到预定义的事件，使用所有找到的事件（智能回退机制）
    if not real_event_ids and mapping:
        print("Using all found event markers:")
        for key, value in mapping.items():
            # 使用英文标签避免中文问题
            real_event_ids[f"Event_{key}"] = value
            print(f"  {key} -> ID: {value}")

    if not real_event_ids:
        print("Error: No valid event markers found")
        return None

    # 保存事件图（使用英文标题）
    plt.figure(figsize=(12, 4))
    mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                        event_id=mapping, first_samp=raw.first_samp)
    plt.title('Event Distribution')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/events_distribution.png', dpi=300, bbox_inches='tight')
    print("Event distribution plot saved")

    # ==========================================
    # 4. 数据分段
    # ==========================================
    print("\nStep 4: Data epoching...")

    # 剔除伪迹标准
    reject_criteria = dict(eeg=100e-6)  # 100微伏

    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=real_event_ids,
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0),
            reject=reject_criteria,
            preload=True
        )

        print(f"Epoching results:")
        for condition in real_event_ids.keys():
            n_epochs = len(epochs[condition])
            print(f"  {condition}: {n_epochs} trials")

    except Exception as e:
        print(f"Error during epoching: {e}")
        # 尝试不使用reject criteria
        try:
            epochs = mne.Epochs(
                raw,
                events,
                event_id=real_event_ids,
                tmin=tmin,
                tmax=tmax,
                baseline=(tmin, 0),
                preload=True
            )
            print(f"Epoching results (without rejection):")
            for condition in real_event_ids.keys():
                n_epochs = len(epochs[condition])
                print(f"  {condition}: {n_epochs} trials")
        except Exception as e2:
            print(f"Epoching failed: {e2}")
            return None

    # ==========================================
    # 5. 叠加平均
    # ==========================================
    print("\nStep 5: Averaging...")

    evoked_dict = {}
    for condition in real_event_ids.keys():
        if len(epochs[condition]) > 0:
            evoked_dict[condition] = epochs[condition].average()
            print(f"{condition} condition averaged ({len(epochs[condition])} trials)")
        else:
            print(f"Warning: No trials for {condition} condition")

    if not evoked_dict:
        print("Error: No valid averaged data")
        return None

    # ==========================================
    # 6. 结果可视化与保存
    # ==========================================
    print("\nStep 6: Generating results...")

    # 定义感兴趣通道(总通道在这定义）
    roi_channels = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    available_picks = [ch for ch in roi_channels if ch in raw.ch_names] #检查是否在原数据中

    if available_picks:
        print(f"Using ROI channels: {available_picks[:5]}...")
        picks = available_picks
    else:
        print("Using all channels")
        picks = 'eeg'

    # 6.1 对比不同条件的ERP波形（使用英文标签）
    plt.figure(figsize=(15, 10)) #前面为宽度，后面为高度，如果要增加显示的通道数，记得增加高度，如(15,15)

    if available_picks:
        # 显示前4个通道
        num_channels_to_plot = 4 #如果要修改显示的通道数，在这里定义数字
        display_channels = available_picks[:num_channels_to_plot] # 在这定义显示的通道
        for i, channel in enumerate(display_channels, 1):
            plt.subplot(2, 2, i) #在这里定义显示图表的几行几列，eg(3，2，i)就是3行2列
            # 存储每个条件的峰值和谷值
            channel_peaks = {}
            channel_troughs = {}
            # for condition, evoked in evoked_dict.items():
            #     if channel in evoked.ch_names:
            #         ch_idx = evoked.ch_names.index(channel)
            #         plt.plot(evoked.times, evoked.data[ch_idx] * 1e6,
            #                  label=condition, linewidth=2)
            # 绘制每个条件的曲线
            for condition, evoked in evoked_dict.items():
                if channel in evoked.ch_names:
                    ch_idx = evoked.ch_names.index(channel)
                    # 获取微伏单位的数据
                    data_microvolts = evoked.data[ch_idx] * 1e6

                    # 找峰值和谷值
                    peak_times, trough_times, peak_indices, trough_indices = find_peaks_and_troughs(
                        evoked.times,
                        data_microvolts,
                        prominence=np.std(data_microvolts) * 1.5,
                        distance=int(len(evoked.times) * 0.1)  # 限制峰值/谷值之间的最小距离
                    )

                    # 绘制曲线
                    plt.plot(evoked.times, data_microvolts,
                             label=condition, linewidth=2)

                    # 标注峰值 (红色虚线)
                    for p_time in peak_times:
                        plt.axvline(x=p_time, color='red', linestyle=':', alpha=0.7, linewidth=1)
                        # 可以在曲线下方或上方添加文字标签，避免遮挡
                        # plt.text(p_time, plt.ylim()[1]*0.9, f'{p_time:.2f}s', color='red', rotation=90, va='top', ha='right', fontsize=8)

                    # 标注谷值 (绿色虚线)
                    for t_time in trough_times:
                        plt.axvline(x=t_time, color='green', linestyle=':', alpha=0.7, linewidth=1)
                        # plt.text(t_time, plt.ylim()[0]*0.9, f'{t_time:.2f}s', color='green', rotation=90, va='bottom', ha='left', fontsize=8)
                    # --- 结束新的标注方式 ---

                    # 存储峰值和谷值信息
                    channel_peaks[condition] = list(zip(peak_times, data_microvolts[peak_indices]))
                    channel_troughs[condition] = list(zip(trough_times, data_microvolts[trough_indices]))
            plt.title(f'Channel {channel}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (μV)')
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Stimulus')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.legend()
            plt.grid(True, alpha=0.3)

            #在终端打印峰值
            print(f"="*50)
            print(f"\nChannel {channel} Peak and Trough Information:")
            for condition in channel_peaks.keys():
                print(f"{condition} Peaks:")
                for time, amplitude in channel_peaks[condition]:
                    print(f"  Time: {time:.3f}s, Amplitude: {amplitude:.2f}μV")
                print(f"{condition} Troughs:")
                for time, amplitude in channel_troughs[condition]:
                    print(f"  Time: {time:.3f}s, Amplitude: {amplitude:.2f}μV")
            print(f"=" * 50)

        plt.suptitle('Stroop ERP - Channel Comparison', fontsize=14)
        plt.tight_layout()
    else:
        # 单个条件图
        condition = list(evoked_dict.keys())[0]
        evoked_dict[condition].plot_joint(title=f"{condition} Condition ERP")

    plt.savefig(f'{output_dir}/erp_waveforms.png', dpi=300, bbox_inches='tight')
    print("ERP waveforms plot saved")

    # 6.2 地形图（使用英文标题）
    # MNE 的 plot_topomap 内部可以处理多时间点的子图布局
    if len(evoked_dict) > 0 and len(raw.ch_names) >= 4:  # 确保有数据且通道足够
        try:
            # 循环遍历所有条件
            for condition_to_plot, evoked_to_plot in evoked_dict.items():
                times_for_topomap = [0.1, 0.2, 0.3, 0.4]  # 示例时间点,可根据erp图自行修改

                fig_topo = evoked_to_plot.plot_topomap(
                    times=times_for_topomap,
                    average=0.05,
                    ch_type='eeg',
                    show=False,
                    cmap='RdBu_r',
                    units='µV'  # 将单位设置为微伏
                )

                fig_topo.suptitle(f'ERP Topomap Sequence - {condition_to_plot}', fontsize=14)
                fig_topo.savefig(f'{output_dir}/erp_topomap_sequence_{condition_to_plot.lower()}.png', dpi=300,
                                 bbox_inches='tight')
                plt.close(fig_topo)
                print(f"Topomap sequence for {condition_to_plot} saved")
        except Exception as e:
            print(f"Topomap generation failed: {e}")

    # 6.3 保存ERP数据
    for condition, evoked in evoked_dict.items():
        # 保存为CSV（使用英文文件名）
        times = evoked.times
        data_microvolt = evoked.data * 1e6  # 转换为微伏

        data_to_save = np.column_stack([times, data_microvolt.T])
        header = 'Time(s),' + ','.join(evoked.ch_names)

        np.savetxt(f'{output_dir}/erp_{condition.lower()}.csv',
                   data_to_save, delimiter=',',
                   header=header, comments='', fmt='%.6f')

        # 保存为fif格式
        # evoked.save(f'{output_dir}/evoked_{condition.lower()}-ave.fif')

        print(f"{condition} condition data saved")

    # ==========================================
    # 7. 统计分析
    # ==========================================
    print("\nStep 7: Statistical analysis...")

    # 计算P300时间窗的平均振幅 (250-450ms)
    time_window = (0.25, 0.45)

    stats_results = {}
    for condition, evoked in evoked_dict.items():
        # 尝试找到Pz通道，如果没有则使用中央区域通道
        target_channels = ['Pz', 'Cz', 'P3', 'P4', 'CPz']
        channel_found = None

        for ch in target_channels:
            if ch in evoked.ch_names:
                channel_found = ch
                break

        if channel_found is None and len(evoked.ch_names) > 0:
            channel_found = evoked.ch_names[0]  # 使用第一个可用通道

        if channel_found:
            channel_idx = evoked.ch_names.index(channel_found)
            data = evoked.data[channel_idx] * 1e6  # 微伏
            times = evoked.times

            # 找到时间窗内的数据点
            time_mask = (times >= time_window[0]) & (times <= time_window[1])
            if np.any(time_mask):
                mean_amplitude = np.mean(data[time_mask])
                max_amplitude = np.max(data[time_mask])
                max_latency = times[time_mask][np.argmax(data[time_mask])]

                stats_results[condition] = {
                    'channel': channel_found,
                    'mean_amplitude': mean_amplitude,
                    'max_amplitude': max_amplitude,
                    'latency': max_latency
                }

                print(f"{condition} ({channel_found}): "
                      f"Mean amplitude = {mean_amplitude:.2f}μV, "
                      f"Max amplitude = {max_amplitude:.2f}μV, "
                      f"Latency = {max_latency * 1000:.1f}ms")

    # 保存统计结果（使用英文）
    with open(f'{output_dir}/statistical_results.txt', 'w', encoding='utf-8') as f:
        f.write("Stroop ERP Statistical Analysis Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Data file: {os.path.basename(file_path)}\n")
        f.write(f"Analysis time: {np.datetime64('now')}\n")
        f.write("P300 time window: 250-450ms\n\n")

        for condition, results in stats_results.items():
            f.write(f"{condition} Condition:\n")
            f.write(f"  Analysis channel: {results['channel']}\n")
            f.write(f"  Mean amplitude: {results['mean_amplitude']:.2f} μV\n")
            f.write(f"  Max amplitude: {results['max_amplitude']:.2f} μV\n")
            f.write(f"  Latency: {results['latency'] * 1000:.1f} ms\n\n")

    print("Statistical analysis completed")
    print("All results saved to", output_dir)

    # 加上时频分析代码
    try:
        time_frequency_analysis(epochs, real_event_ids, output_dir)
    except Exception as e:
        print(f"时频分析出错: {e}")

    return evoked_dict, stats_results


def find_correct_file_path():
    """根据你的项目结构找到正确的文件路径"""

    # 尝试多个可能的路径
    possible_paths = [
        # 根据你的项目结构
        "D:/AllLanProject/python/DataPro1/Acquisition 190.cdt",
        "Acquisition 190.cdt",
        "data/Acquisition 190.cdt",
        "../Acquisition 190.cdt",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found file: {path}")
            return path

    # 如果以上路径都不存在，搜索整个项目
    project_root = "D:/AllLanProject/python/DataPro1"
    if os.path.exists(project_root):
        for root, dirs, files in os.walk(project_root):
            if "Acquisition 190.cdt" in files:
                found_path = os.path.join(root, "Acquisition 190.cdt")
                print(f"Found file: {found_path}")
                return found_path

    print("File not found, please check file location")
    return None

# 注：需下载h5io库：pip install h5io
def time_frequency_analysis(epochs, real_event_ids, output_dir='results'):
    """
    执行时频分析

    参数:
    - epochs: MNE epochs对象
    - real_event_ids: 事件标签字典
    - output_dir: 结果保存目录
    """
    import mne
    import numpy as np
    import matplotlib.pyplot as plt

    print("\n" + "="*50)
    print("时频分析 (Wavelet Transform)")
    print("="*50)

    # 设置频率范围和循环数
    freqs = np.logspace(*np.log10([4, 50]), num=30)  # 4-50 Hz
    n_cycles = freqs / 2.0  # 每个频率的小波周期数

    # 与上面erp显示相同
    # 使用 epochs.ch_names 来确保只选择存在的通道
    roi_channels = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4']
    available_channels = [ch for ch in roi_channels if ch in epochs.ch_names]

    if not available_channels:
        available_channels = epochs.ch_names[:5] # Fallback to first 5 available if ROI not found

    # 为每个条件进行时频分析
    for condition_name, condition_id in real_event_ids.items(): # 明确使用 condition_name
        # 确保该条件存在有效的epoch
        if condition_name not in epochs.event_id:
             print(f"警告: {condition_name} 未在 epoch 对象中找到，跳过。")
             continue

        # 获取该条件下的所有epoch
        epochs_for_condition = epochs[condition_name]
        if len(epochs_for_condition) == 0:
            print(f"警告: {condition_name} 条件没有有效的 epoch，跳过。")
            continue

        print(f"\n分析条件: {condition_name}")

        # 对每个选定的通道进行时频分析
        # 只取前3个通道进行绘图，以免生成过多图片
        for channel in available_channels[:3]: #在这定义了显示的通道数
            print(f"  通道: {channel}")

            # 提取单个通道的epochs数据进行时频分析
            # 确保 pick_channels 选中的通道是实际存在的
            if channel not in epochs_for_condition.ch_names:
                print(f"  警告: 通道 {channel} 不存在于 {condition_name} 的 epochs 中，跳过。")
                continue

            # 创建一个新的epochs对象，只包含当前通道
            epochs_single_channel = epochs_for_condition.copy().pick_channels([channel])

            print(f"DEBUG: Type of mne: {type(mne)}")
            print(f"DEBUG: Type of mne.time_frequency: {type(mne.time_frequency)}")
            print(f"DEBUG: Is mne.time_frequency.tfr_morlet callable? {callable(mne.time_frequency.tfr_morlet)}")
            print(f"DEBUG: Type of epochs_single_channel: {type(epochs_single_channel)}")
            print(f"DEBUG: Number of epochs in epochs_single_channel: {len(epochs_single_channel)}")  # 确保这里不是0或None

            power = mne.time_frequency.tfr_morlet(
                epochs_single_channel,  # 使用单通道epochs
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                decim=3,
                n_jobs=1
            )
            # 执行小波变换
            power = mne.time_frequency.tfr_morlet(
                epochs_single_channel, # 使用单通道epochs
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                decim=3,
                n_jobs=1
            )

            # 保存时频结果
            file_name_prefix = f'{output_dir}/tfr_{condition_name.lower().replace(" ", "_")}_{channel.lower()}'
            # power.save(f'{file_name_prefix}-tfr.h5', overwrite=True)

            # 绘制时频图
            fig, ax = plt.subplots(figsize=(10, 6))
            power.plot(
                picks=0,  # 确保这里是索引0，因为power对象现在只有一个通道
                baseline=(-0.2, 0),  # 基线校正
                mode='percent',  # 以百分比变化显示
                title=f'{condition_name} - {channel}',
                show=False,
                axes=ax,
                colorbar=True,
                cmap='RdBu_r'
            )
            plt.tight_layout()
            plt.savefig(f'{file_name_prefix}.png', dpi=300)
            plt.close()

        print(f"  ✓ {condition_name} 条件时频分析完成")

    print("\n时频分析全部完成")


def find_peaks_and_troughs(times, data, prominence=None, distance=None):
    """
    使用 scipy 找到信号的峰值和谷值

    参数:
    - times: 时间序列
    - data: 信号数据
    - prominence: 峰值/谷值显著性阈值
    - distance: 两个峰值/谷值之间的最小距离

    返回:
    - peaks: 峰值的时间点和索引
    - troughs: 谷值的时间点和索引
    """
    from scipy.signal import find_peaks

    # 如果没有指定 prominence，自动设置一个合理的值
    if prominence is None:
        prominence = np.std(data) * 1.5

    # 找峰值
    peaks, _ = find_peaks(data, prominence=prominence, distance=distance)
    # 找谷值（对数据取负，然后找峰值）
    troughs, _ = find_peaks(-data, prominence=prominence, distance=distance)

    # 转换为时间点
    peak_times = times[peaks]
    trough_times = times[troughs]

    return peak_times, trough_times, peaks, troughs

# ==========================================
# 主程序执行
# ==========================================
if __name__ == "__main__":
    # 设置控制台编码为UTF-8
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("Current working directory:", os.getcwd())
    print("Script location:", os.path.abspath(__file__))

    # 自动查找正确的文件路径
    file_path = find_correct_file_path()

    if file_path is None:
        # 如果自动查找失败，使用硬编码路径
        file_path = "D:/AllLanProject/python/DataPro1/Acquisition 190.cdt"
        print(f"Using hardcoded path: {file_path}")

    ### 可以根据下面的输出来得到对应的内部ID
    # import mne
    #
    # file_path = "D:/AllLanProject/python/DataPro1/Acquisition 190.cdt"  # 替换成你的文件路径
    # raw = mne.io.read_raw_curry(file_path, preload=True)
    # events, mapping = mne.events_from_annotations(raw)
    # print("Actual event markers:", mapping)


    ###此处为显示的图表的显示定义
    # 事件标记定义
    event_dict = {
        'NoGo': 1,  # 因为marker值为11
        'Go': 2  # marker值为21
    }

    # 如果标准标记不工作，尝试不同的事件标记方案
    alternative_event_dicts = [
        event_dict,
        # {'Congruent': 11, 'Incongruent': 22, 'Neutral': 33},
        # {'Congruent': 101, 'Incongruent': 102, 'Neutral': 103},
        # {'Congruent': 201, 'Incongruent': 202, 'Neutral': 203},
        # {'Condition1': 1, 'Condition2': 2, 'Condition3': 3}
    ]

    success = False
    for i, test_dict in enumerate(alternative_event_dicts):
        print(f"\nTrying event marker scheme {i + 1}: {test_dict}")
        try:
            results = stroop_erp_analysis(file_path, test_dict)
            if results is not None:
                success = True
                print("Analysis completed successfully!")
                break
            else:
                print("This scheme failed, trying next...")
        except Exception as e:
            print(f"Error: {e}")
            continue

    # 或者根据下面的输出也能得到
    if not success:
        print("\nAll event marker schemes failed, please check event markers manually")
        print("Run the following code to check actual event markers:")
        print("""
import mne
file_path = "D:/code/Code_ecnu_projects/LLM_Learn/Test_mine/Homework/NaoDian/data/Acquisition 190.cdt"
raw = mne.io.read_raw_curry(file_path, preload=True)
events, mapping = mne.events_from_annotations(raw)
print("Actual event markers:", mapping)
        """)

    # 显示图表
    plt.show()
