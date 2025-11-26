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
        project_dir = "D:/code/Code_ecnu_projects/LLM_Learn"
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

    # 获取事件标记
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

    # 如果没有找到预定义的事件，使用所有找到的事件
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

    # 定义感兴趣通道
    roi_channels = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    available_picks = [ch for ch in roi_channels if ch in raw.ch_names]

    if available_picks:
        print(f"Using ROI channels: {available_picks[:5]}...")
        picks = available_picks
    else:
        print("Using all channels")
        picks = 'eeg'

    # 6.1 对比不同条件的ERP波形（使用英文标签）
    plt.figure(figsize=(18, 12))

    if available_picks:
        # 显示前6个通道
        display_channels = available_picks[:6]
        for i, channel in enumerate(display_channels, 1):
            plt.subplot(3, 2, i)
            for condition, evoked in evoked_dict.items():
                if channel in evoked.ch_names:
                    ch_idx = evoked.ch_names.index(channel)
                    plt.plot(evoked.times, evoked.data[ch_idx] * 1e6,
                             label=condition, linewidth=2)
            plt.title(f'Channel {channel}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (μV)')
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Stimulus')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.suptitle('Stroop ERP - Channel Comparison', fontsize=14)
        plt.tight_layout()
    else:
        # 单个条件图
        condition = list(evoked_dict.keys())[0]
        evoked_dict[condition].plot_joint(title=f"{condition} Condition ERP")

    plt.savefig(f'{output_dir}/erp_waveforms.png', dpi=300, bbox_inches='tight')
    print("ERP waveforms plot saved")

    # 6.2 地形图（使用英文标题）
    if len(raw.ch_names) >= 4:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.ravel()

            times = [0.1, 0.2, 0.3, 0.4]  # 不同时间点
            for i, time in enumerate(times):
                if i < len(axes):
                    condition = list(evoked_dict.keys())[0]
                    evoked_dict[condition].plot_topomap(
                        times=time, axes=axes[i], show=False,
                        time_unit='s', size=3
                    )
                    axes[i].set_title(f'{time * 1000:.0f}ms')

            plt.suptitle('ERP Topomap Sequence')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/topomap_sequence.png', dpi=300, bbox_inches='tight')
            print("Topomap sequence saved")
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
        evoked.save(f'{output_dir}/evoked_{condition.lower()}-ave.fif')

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

    return evoked_dict, stats_results


def find_correct_file_path():
    """根据你的项目结构找到正确的文件路径"""

    # 尝试多个可能的路径
    possible_paths = [
        # 根据你的项目结构
        "D:/code/Code_ecnu_projects/LLM_Learn/Test_mine/Homework/NaoDian/data/Acquisition 190.cdt",
        "Acquisition 190.cdt",
        "data/Acquisition 190.cdt",
        "../data/Acquisition 190.cdt",
        "../../data/Acquisition 190.cdt",
        "Test_mine/Homework/NaoDian/data/Acquisition 190.cdt",
        "Test mine/Homework/NaoDian/data/Result/Acquisition 190.cdt",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found file: {path}")
            return path

    # 如果以上路径都不存在，搜索整个项目
    project_root = "D:/code/Code_ecnu_projects/LLM_Learn"
    if os.path.exists(project_root):
        for root, dirs, files in os.walk(project_root):
            if "Acquisition 190.cdt" in files:
                found_path = os.path.join(root, "Acquisition 190.cdt")
                print(f"Found file: {found_path}")
                return found_path

    print("File not found, please check file location")
    return None


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
        file_path = "D:/code/Code_ecnu_projects/LLM_Learn/Test_mine/Homework/NaoDian/data/Acquisition 190.cdt"
        print(f"Using hardcoded path: {file_path}")

    # 事件标记定义 - 使用英文避免编码问题
    event_dict = {
        'Congruent': 1,  # 一致条件
        'Incongruent': 2,  # 不一致条件
        'Neutral': 3  # 中性条件
    }

    # 如果标准标记不工作，尝试不同的事件标记方案
    alternative_event_dicts = [
        event_dict,
        {'Congruent': 11, 'Incongruent': 22, 'Neutral': 33},
        {'Congruent': 101, 'Incongruent': 102, 'Neutral': 103},
        {'Congruent': 201, 'Incongruent': 202, 'Neutral': 203},
        {'Condition1': 1, 'Condition2': 2, 'Condition3': 3}
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