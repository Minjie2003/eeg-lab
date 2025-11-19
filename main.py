"""
Go/NoGo 脑电数据预处理脚本
使用MNE-Python进行EEG数据的预处理
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings('ignore')


def check_dependencies():
    """检查必要的依赖包是否已安装"""
    required_packages = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'mne': 'mne',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn'
    }

    missing_packages = []

    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print("=" * 60)
        print("❌ 缺少必要的依赖包:")
        print("=" * 60)
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n请运行以下命令安装:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\n或者:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        return False

    return True

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EEGPreprocessor:
    """脑电数据预处理类"""

    def __init__(self, data_path, subject_name='Acquisition 190'):
        """
        初始化预处理器

        Parameters:
        -----------
        data_path : str
            数据文件所在目录
        subject_name : str
            被试名称/文件前缀
        """
        self.data_path = data_path
        self.subject_name = subject_name
        self.raw = None
        self.epochs = None
        self.ica = None
        self.evoked_dict = {}

    def load_data(self):
        """加载Curry格式的脑电数据"""
        print("=" * 60)
        print("步骤 1: 加载数据")
        print("=" * 60)

        # 构建文件路径
        cdt_file = os.path.join(self.data_path, f"{self.subject_name}.cdt")

        try:
            # 读取Curry数据
            self.raw = mne.io.read_raw_curry(cdt_file, preload=True, verbose=True)
            print(f"\n✓ 成功加载数据: {cdt_file}")
            print(f"  - 采样率: {self.raw.info['sfreq']} Hz")
            print(f"  - 通道数: {len(self.raw.ch_names)}")
            print(f"  - 数据长度: {self.raw.times[-1]:.2f} 秒")
            print(f"  - 通道名称: {self.raw.ch_names}")

            # 设置特殊通道类型
            self._set_channel_types()

        except Exception as e:
            print(f"✗ 加载数据失败: {e}")
            print("\n尝试其他读取方法...")
            raise

        return self.raw

    def _set_channel_types(self):
        """设置EOG和Stim通道类型"""
        print("\n设置通道类型...")
        channel_type_mapping = {}

        # 设置EOG通道
        if 'VEOG' in self.raw.ch_names:
            channel_type_mapping['VEOG'] = 'eog'
        if 'HEOG' in self.raw.ch_names:
            channel_type_mapping['HEOG'] = 'eog'

        # 设置Trigger/Stim通道
        if 'Trigger' in self.raw.ch_names:
            channel_type_mapping['Trigger'] = 'stim'

        if channel_type_mapping:
            self.raw.set_channel_types(channel_type_mapping)
            print(f"✓ 已设置通道类型: {channel_type_mapping}")
        else:
            print("  未发现需要特殊设置的通道")

        return self.raw

    def set_montage(self, montage_name='standard_1020'):
        """设置电极位置"""
        print("\n" + "=" * 60)
        print("步骤 2: 设置电极位置")
        print("=" * 60)

        # 检查是否有特殊电极（非标准10-20系统）
        problematic_channels = ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2']
        channels_to_remove = [ch for ch in problematic_channels if ch in self.raw.ch_names]

        if channels_to_remove:
            print(f"检测到非标准电极: {channels_to_remove}")
            print("这些电极会导致可视化问题，将自动移除")
            self.raw.drop_channels(channels_to_remove)
            print(f"✓ 已移除问题电极，保留 {len(self.raw.ch_names)} 个通道")

        try:
            montage = mne.channels.make_standard_montage(montage_name)
            self.raw.set_montage(montage, on_missing='warn')
            print(f"✓ 成功设置电极位置: {montage_name}")
        except Exception as e:
            print(f"⚠ 设置电极位置时出现警告: {e}")
            print("  将继续进行，但可能影响后续的可视化")

    def filter_data(self, l_freq=0.5, h_freq=40):
        """
        滤波

        Parameters:
        -----------
        l_freq : float
            高通滤波截止频率（Hz）
        h_freq : float
            低通滤波截止频率（Hz）
        """
        print("\n" + "=" * 60)
        print("步骤 3: 滤波")
        print("=" * 60)

        print(f"应用带通滤波: {l_freq}-{h_freq} Hz")
        self.raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        print("✓ 滤波完成")

        # 绘制功率谱密度
        fig = self.raw.plot_psd(fmax=50, average=True, spatial_colors=False)
        fig.suptitle('滤波后的功率谱密度')
        plt.tight_layout()

    def detect_bad_channels(self, method='auto'):
        """
        检测坏导联

        Parameters:
        -----------
        method : str
            'auto' 自动检测或 'manual' 手动标记
        """
        print("\n" + "=" * 60)
        print("步骤 4: 检测坏导联")
        print("=" * 60)

        if method == 'auto':
            # 只检测EEG通道（不检测EOG、Stim等通道）
            eeg_picks = mne.pick_types(self.raw.info, eeg=True, eog=False, stim=False, exclude=[])

            if len(eeg_picks) == 0:
                print("⚠ 没有找到EEG通道")
                return []

            data = self.raw.get_data(picks=eeg_picks)
            eeg_ch_names = [self.raw.ch_names[i] for i in eeg_picks]

            # 基于标准差检测坏导联
            stds = np.std(data, axis=1)
            threshold_high = 3 * np.median(stds)
            threshold_low = np.median(stds) / 3

            bad_channels = [eeg_ch_names[i] for i, std in enumerate(stds)
                          if std > threshold_high or std < threshold_low]

            if bad_channels:
                self.raw.info['bads'] = bad_channels
                print(f"✓ 检测到坏导联: {bad_channels}")
            else:
                print("✓ 未检测到明显的坏导联")

        return self.raw.info['bads']

    def set_reference(self, ref_channels='average'):
        """
        重参考

        Parameters:
        -----------
        ref_channels : str or list
            'average' 平均参考, 或指定参考电极列表
        """
        print("\n" + "=" * 60)
        print("步骤 5: 重参考")
        print("=" * 60)

        if ref_channels == 'average':
            self.raw.set_eeg_reference('average', projection=True)
            print("✓ 设置为平均参考")
        else:
            self.raw.set_eeg_reference(ref_channels)
            print(f"✓ 设置参考电极: {ref_channels}")

        self.raw.apply_proj()

    def run_ica(self, n_components=15, method='fastica', random_state=42):
        """
        运行ICA去除眼电等伪迹

        Parameters:
        -----------
        n_components : int
            ICA成分数量
        method : str
            ICA算法 ('fastica', 'infomax', 'picard')
        random_state : int
            随机种子
        """
        print("\n" + "=" * 60)
        print("步骤 6: ICA去除伪迹")
        print("=" * 60)

        print(f"运行ICA (成分数: {n_components}, 方法: {method})")

        # 尝试不同的ICA方法
        methods_to_try = [method, 'picard', 'infomax']

        for m in methods_to_try:
            try:
                # 创建ICA对象
                self.ica = ICA(n_components=n_components, method=m,
                              random_state=random_state, max_iter='auto')

                # 拟合ICA
                self.ica.fit(self.raw, verbose=False)
                print(f"✓ ICA拟合完成 (使用方法: {m})")
                break

            except ImportError as e:
                if m == methods_to_try[-1]:
                    print(f"\n✗ 所有ICA方法都失败")
                    print(f"错误信息: {e}")
                    print("\n请确保已安装scikit-learn:")
                    print("  pip install scikit-learn")
                    raise
                else:
                    print(f"⚠ 方法 {m} 不可用，尝试 {methods_to_try[methods_to_try.index(m)+1]}...")
                    continue
            except Exception as e:
                print(f"✗ ICA拟合失败: {e}")
                raise

        # 可视化ICA成分
        print("\n绘制ICA成分...")
        try:
            self.ica.plot_sources(self.raw, show_scrollbars=False)
        except Exception as e:
            print(f"⚠ 绘制ICA时间序列时出现问题: {e}")
            print("  将跳过此可视化")

        try:
            self.ica.plot_components(inst=self.raw)
        except ValueError as e:
            if "overlapping positions" in str(e):
                print(f"⚠ 无法绘制ICA成分地形图（电极位置重叠）")
                print("  将使用备选可视化方法...")
                try:
                    # 尝试使用不需要位置信息的可视化
                    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
                    fig.suptitle('ICA成分', fontsize=16)
                    for idx, ax in enumerate(axes.flat):
                        if idx < self.ica.n_components_:
                            ax.plot(self.ica.get_components()[:, idx])
                            ax.set_title(f'IC{idx}')
                            ax.set_xlabel('Channel')
                            ax.set_ylabel('Weight')
                        else:
                            ax.axis('off')
                    plt.tight_layout()
                    print("✓ 使用简化的ICA成分可视化")
                except Exception as e2:
                    print(f"⚠ 备选可视化也失败: {e2}")
            else:
                raise
        except Exception as e:
            print(f"⚠ 绘制ICA成分时出现问题: {e}")
            print("  将跳过此可视化")

        # 自动检测眼电成分（如果有EOG通道）
        eog_channels = mne.pick_types(self.raw.info, eog=True)
        if len(eog_channels) > 0:
            eog_ch_names = [self.raw.ch_names[i] for i in eog_channels]
            print(f"\n找到EOG通道: {eog_ch_names}")
            print("尝试自动检测眼电成分...")

            try:
                # 明确指定EOG通道进行检测
                eog_indices, eog_scores = self.ica.find_bads_eog(
                    self.raw,
                    ch_name=eog_ch_names,
                    threshold=3.0,
                    verbose=False
                )

                if len(eog_indices) > 0:
                    self.ica.exclude = eog_indices
                    print(f"✓ 自动检测到眼电成分: {eog_indices}")
                    # 显示每个成分的相关系数
                    for idx in eog_indices:
                        print(f"  IC{idx}: 相关系数 = {eog_scores[idx]:.3f}")
                else:
                    print("⚠ 未检测到明显的眼电成分（阈值: 3.0）")
                    print("  建议手动查看ICA成分图")

            except Exception as e:
                print(f"⚠ 自动检测失败: {e}")
                print("  需要手动选择伪迹成分")
        else:
            print("⚠ 未找到EOG通道，需要手动选择伪迹成分")
            print("  请查看ICA成分图，手动指定要排除的成分索引")

        return self.ica

    def apply_ica(self, exclude_components=None):
        """
        应用ICA，去除指定成分

        Parameters:
        -----------
        exclude_components : list
            要排除的ICA成分索引列表
        """
        if exclude_components is not None:
            self.ica.exclude = exclude_components
            print(f"\n设置排除的ICA成分: {exclude_components}")

        print("\n应用ICA去除伪迹...")
        self.raw = self.ica.apply(self.raw.copy())
        print("✓ ICA应用完成")

    def find_events(self):
        """查找事件标记"""
        print("\n" + "=" * 60)
        print("步骤 7: 查找事件标记")
        print("=" * 60)

        events = None

        # 方法1: 尝试从Trigger通道读取
        if 'Trigger' in self.raw.ch_names:
            print("从Trigger通道查找事件...")
            try:
                events = mne.find_events(self.raw, stim_channel='Trigger',
                                        verbose=True, min_duration=0.002)
                if len(events) > 0:
                    print(f"✓ 从Trigger通道找到 {len(events)} 个事件")
            except Exception as e:
                print(f"⚠ 从Trigger通道读取失败: {e}")

        # 方法2: 如果方法1失败，尝试从stim类型通道读取
        if events is None or len(events) == 0:
            stim_channels = mne.pick_types(self.raw.info, stim=True)
            if len(stim_channels) > 0:
                stim_ch_name = self.raw.ch_names[stim_channels[0]]
                print(f"从stim通道 {stim_ch_name} 查找事件...")
                try:
                    events = mne.find_events(self.raw, stim_channel=stim_ch_name,
                                            verbose=True, min_duration=0.002)
                    if len(events) > 0:
                        print(f"✓ 从 {stim_ch_name} 找到 {len(events)} 个事件")
                except Exception as e:
                    print(f"⚠ 从 {stim_ch_name} 读取失败: {e}")

        # 方法3: 尝试从annotations提取
        if (events is None or len(events) == 0) and len(self.raw.annotations) > 0:
            print("从annotations提取事件...")
            try:
                events, event_id = mne.events_from_annotations(self.raw, verbose=True)
                if len(events) > 0:
                    print(f"✓ 从annotations提取到 {len(events)} 个事件")
                    print(f"  事件ID映射: {event_id}")
            except Exception as e:
                print(f"⚠ 从annotations提取失败: {e}")

        # 显示结果
        if events is not None and len(events) > 0:
            print(f"\n✓ 总共找到 {len(events)} 个事件")

            # 获取唯一的事件类型
            unique_events = np.unique(events[:, 2])
            print(f"  事件类型: {unique_events}")

            # 统计每种事件的数量
            for evt in unique_events:
                count = np.sum(events[:, 2] == evt)
                print(f"  事件 {evt}: {count} 次")

            # 绘制事件
            try:
                fig = mne.viz.plot_events(events, sfreq=self.raw.info['sfreq'],
                                         first_samp=self.raw.first_samp)
                fig.suptitle('事件标记时间线')
            except Exception as e:
                print(f"⚠ 绘制事件图失败: {e}")
        else:
            print("\n✗ 未找到任何事件标记")
            print("  可能的原因:")
            print("  1. Trigger通道没有记录事件")
            print("  2. 事件编码方式不同")
            print("  3. 需要从其他来源导入事件")
            events = np.array([])

        return events

    def create_epochs(self, events, event_id=None, tmin=-0.2, tmax=0.8,
                     baseline=(None, 0), reject=None):
        """
        创建epochs

        Parameters:
        -----------
        events : array
            事件数组
        event_id : dict
            事件ID字典，例如 {'Go': 11, 'NoGo': 21}
        tmin : float
            epoch开始时间（相对于事件，秒）
        tmax : float
            epoch结束时间（相对于事件，秒）
        baseline : tuple
            基线校正时间窗口
        reject : dict
            拒绝标准，例如 {'eeg': 100e-6}
        """
        print("\n" + "=" * 60)
        print("步骤 8: 创建Epochs")
        print("=" * 60)

        if reject is None:
            reject = {'eeg': 100e-6}  # 100 µV

        print(f"\nEpoch参数:")
        print(f"  事件标记: {event_id}")
        print(f"  时间窗口: {tmin} 到 {tmax} 秒")
        print(f"  基线: {baseline}")
        print(f"  拒绝标准: {reject}")

        self.epochs = mne.Epochs(self.raw, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, baseline=baseline,
                                reject=reject, preload=True, verbose=False)

        print(f"\n✓ 创建了 {len(self.epochs)} 个epochs")
        print(f"  各条件的epoch数量:")
        for event_name in self.epochs.event_id.keys():
            count = len(self.epochs[event_name])
            print(f"    {event_name}: {count}")

        # 绘制epochs图像
        try:
            self.epochs.plot(n_epochs=5, n_channels=30, scalings='auto')
        except Exception as e:
            print(f"⚠ 绘制epochs失败: {e}")

        return self.epochs

    def compute_erp(self):
        """计算事件相关电位（ERP）"""
        print("\n" + "=" * 60)
        print("步骤 9: 计算ERP叠加平均")
        print("=" * 60)

        if self.epochs is None:
            print("✗ 请先创建epochs")
            return None

        # 对每个条件计算叠加平均
        self.evoked_dict = {}
        for condition in self.epochs.event_id.keys():
            evoked = self.epochs[condition].average()
            self.evoked_dict[condition] = evoked
            print(f"✓ {condition}: 叠加平均了 {len(self.epochs[condition])} 个trials")

        return self.evoked_dict

    def plot_erp(self, picks=None, show_difference_wave=True):
        """
        绘制事件相关电位（ERP）

        Parameters:
        -----------
        picks : str or list
            要绘制的电极，如 'eeg' 或 ['Fz', 'Cz', 'Pz']
        show_difference_wave : bool
            是否显示差异波（NoGo - Go）
        """
        print("\n" + "=" * 60)
        print("步骤 10: 绘制ERP波形")
        print("=" * 60)

        if not self.evoked_dict:
            print("✗ 请先计算ERP")
            return

        # 1. 绘制所有条件的ERP对比（叠加图）
        print("\n绘制条件对比图...")
        colors = {'Go': 'blue', 'NoGo': 'red', 'Difference': 'green'}

        try:
            fig = mne.viz.plot_compare_evokeds(
                self.evoked_dict,
                picks=picks if picks else 'eeg',
                combine='mean',
                colors=[colors.get(k, 'black') for k in self.evoked_dict.keys()],
                title='Go vs NoGo ERP对比 (所有电极平均)'
            )
            print("✓ ERP对比波形图绘制完成")
        except Exception as e:
            print(f"⚠ 绘制ERP波形时出现问题: {e}")

        # 2. 绘制典型电极的波形
        print("\n绘制典型电极波形...")
        typical_channels = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz']
        available_channels = [ch for ch in typical_channels
                            if ch in self.evoked_dict[list(self.evoked_dict.keys())[0]].ch_names]

        if available_channels:
            n_channels = len(available_channels)
            fig, axes = plt.subplots(1, n_channels, figsize=(5*n_channels, 4))
            if n_channels == 1:
                axes = [axes]

            for ax, ch in zip(axes, available_channels):
                for condition, evoked in self.evoked_dict.items():
                    evoked_ch = evoked.copy().pick_channels([ch])
                    times = evoked_ch.times * 1000  # 转换为毫秒
                    data = evoked_ch.data[0] * 1e6  # 转换为微伏
                    ax.plot(times, data, label=condition,
                           color=colors.get(condition, 'black'), linewidth=2)

                ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
                ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
                ax.set_xlabel('时间 (ms)', fontsize=12)
                ax.set_ylabel('电压 (μV)', fontsize=12)
                ax.set_title(f'{ch}电极', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            print(f"✓ 绘制了 {available_channels} 电极的波形")
        else:
            print(f"⚠ 未找到典型电极 {typical_channels}")

        # 3. 如果有Go和NoGo条件，计算并绘制差异波
        if show_difference_wave and 'Go' in self.evoked_dict and 'NoGo' in self.evoked_dict:
            print("\n计算差异波 (NoGo - Go)...")
            try:
                diff_wave = mne.combine_evoked([self.evoked_dict['NoGo'],
                                               self.evoked_dict['Go']],
                                              weights=[1, -1])

                # 绘制差异波
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                times = diff_wave.times * 1000
                data = diff_wave.data.mean(axis=0) * 1e6  # 所有电极平均

                ax.plot(times, data, color='green', linewidth=2, label='NoGo - Go')
                ax.fill_between(times, 0, data, where=(data > 0),
                               color='green', alpha=0.3, label='NoGo > Go')
                ax.fill_between(times, 0, data, where=(data < 0),
                               color='red', alpha=0.3, label='Go > NoGo')

                ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
                ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
                ax.set_xlabel('时间 (ms)', fontsize=14)
                ax.set_ylabel('电压差异 (μV)', fontsize=14)
                ax.set_title('差异波: NoGo - Go (所有电极平均)', fontsize=16, fontweight='bold')
                ax.legend(loc='best', fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                print("✓ 差异波绘制完成")

                # 保存差异波到字典
                self.evoked_dict['Difference'] = diff_wave

            except Exception as e:
                print(f"⚠ 计算差异波失败: {e}")

        # 4. 绘制地形图
        print("\n尝试绘制地形图...")
        for condition, evoked in self.evoked_dict.items():
            try:
                times = np.linspace(0, 0.6, 7)  # 0-600ms，每100ms一个时间点
                fig = evoked.plot_topomap(times=times,
                                         title=f'{condition} - 地形图',
                                         time_unit='s',
                                         size=3)
                print(f"✓ {condition} 地形图绘制完成")
            except ValueError as e:
                if "overlapping positions" in str(e):
                    print(f"⚠ {condition} 地形图无法绘制（电极位置重叠）")
                else:
                    print(f"⚠ {condition} 地形图绘制失败: {e}")
            except Exception as e:
                print(f"⚠ {condition} 地形图绘制失败: {e}")

        print("\n✓ ERP分析完成")

    def analyze_erp_components(self):
        """分析ERP成分（如P300, N200等）"""
        print("\n" + "=" * 60)
        print("步骤 11: ERP成分分析")
        print("=" * 60)

        if not self.evoked_dict:
            print("✗ 请先计算ERP")
            return

        # 选择中央电极进行成分分析
        roi_channels = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz']

        for condition, evoked in self.evoked_dict.items():
            if condition == 'Difference':
                continue

            print(f"\n{condition} 条件:")
            available_roi = [ch for ch in roi_channels if ch in evoked.ch_names]

            if not available_roi:
                print("  ⚠ 未找到ROI电极")
                continue

            # 提取ROI平均数据
            evoked_roi = evoked.copy().pick_channels(available_roi)
            data = evoked_roi.data.mean(axis=0) * 1e6  # 转换为微伏
            times = evoked_roi.times * 1000  # 转换为毫秒

            # 寻找N200成分 (150-250ms, 负波)
            n200_window = (times >= 150) & (times <= 250)
            if np.any(n200_window):
                n200_idx = np.argmin(data[n200_window])
                n200_time = times[n200_window][n200_idx]
                n200_amp = data[n200_window][n200_idx]
                print(f"  N200: {n200_time:.1f} ms, {n200_amp:.2f} μV")

            # 寻找P300成分 (250-500ms, 正波)
            p300_window = (times >= 250) & (times <= 500)
            if np.any(p300_window):
                p300_idx = np.argmax(data[p300_window])
                p300_time = times[p300_window][p300_idx]
                p300_amp = data[p300_window][p300_idx]
                print(f"  P300: {p300_time:.1f} ms, {p300_amp:.2f} μV")

    def save_preprocessed_data(self, output_dir='preprocessed_data'):
        """保存预处理后的数据"""
        print("\n" + "=" * 60)
        print("步骤 12: 保存数据")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        # 保存原始数据
        raw_fname = os.path.join(output_dir, f'{self.subject_name}_raw_preprocessed-raw.fif')
        self.raw.save(raw_fname, overwrite=True)
        print(f"✓ 保存预处理后的raw数据: {raw_fname}")

        # 保存epochs
        if self.epochs is not None:
            epochs_fname = os.path.join(output_dir, f'{self.subject_name}_epochs-epo.fif')
            self.epochs.save(epochs_fname, overwrite=True)
            print(f"✓ 保存epochs数据: {epochs_fname}")

        # 保存ICA
        if self.ica is not None:
            ica_fname = os.path.join(output_dir, f'{self.subject_name}_ica.fif')
            self.ica.save(ica_fname, overwrite=True)
            print(f"✓ 保存ICA: {ica_fname}")

        # 保存ERP数据
        if self.evoked_dict:
            for condition, evoked in self.evoked_dict.items():
                evoked_fname = os.path.join(output_dir,
                                           f'{self.subject_name}_{condition}-ave.fif')
                evoked.save(evoked_fname, overwrite=True)
            print(f"✓ 保存ERP数据: {list(self.evoked_dict.keys())}")

        print(f"\n所有数据已保存到: {output_dir}")


def main():
    """主函数"""
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "Go/NoGo EEG 数据预处理" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # 检查依赖包
    if not check_dependencies():
        sys.exit(1)

    # ==================== 设置参数 ====================
    DATA_PATH = 'raw_data'  # 数据文件所在目录
    SUBJECT_NAME = 'Acquisition 190'  # 文件名前缀

    # Go/NoGo实验的事件标记
    EVENT_ID = {
        'Go': 11,      # Go试次的标记
        'NoGo': 21     # NoGo试次的标记
    }
    # =================================================

    # 创建预处理器
    preprocessor = EEGPreprocessor(DATA_PATH, SUBJECT_NAME)

    try:
        # 1. 加载数据
        raw = preprocessor.load_data()

        # 2. 设置电极位置
        preprocessor.set_montage()

        # 3. 滤波
        preprocessor.filter_data(l_freq=0.5, h_freq=40)

        # 4. 检测坏导联
        preprocessor.detect_bad_channels()

        # 5. 重参考
        preprocessor.set_reference('average')

        # 6. 运行ICA
        ica = preprocessor.run_ica(n_components=15)

        # 7. 应用ICA
        if len(ica.exclude) > 0:
            preprocessor.apply_ica()
            print(f"\n✓ 已自动去除ICA成分: {ica.exclude}")
        else:
            print("\n⚠ 未检测到明显的伪迹成分")
            response = input("是否继续不应用ICA？(y/n): ")
            if response.lower() != 'y':
                print("程序终止。请检查ICA成分后手动指定要排除的成分。")
                print("可以修改代码中的: preprocessor.apply_ica(exclude_components=[0, 1])")
                return

        # 8. 查找事件
        events = preprocessor.find_events()

        if len(events) > 0:
            # 9. 创建epochs
            epochs = preprocessor.create_epochs(
                events,
                event_id=EVENT_ID,  # 使用定义的Go/NoGo标记
                tmin=-0.2,          # epoch开始时间（秒）
                tmax=0.8,           # epoch结束时间（秒）
                baseline=(None, 0), # 基线校正
                reject={'eeg': 100e-6}  # 拒绝标准（100 µV）
            )

            # 10. 计算ERP叠加平均
            evoked_dict = preprocessor.compute_erp()

            # 11. 绘制ERP波形（包括差异波）
            preprocessor.plot_erp(show_difference_wave=True)

            # 12. 分析ERP成分
            preprocessor.analyze_erp_components()

        else:
            print("\n⚠ 由于没有找到事件标记，跳过epoch创建和ERP分析")

        # 13. 保存数据
        preprocessor.save_preprocessed_data()

        print("\n" + "=" * 60)
        print("✓✓✓ 预处理完成！✓✓✓")
        print("=" * 60)
        print("\n生成的分析结果包括:")
        print("1. ✓ Go和NoGo条件的ERP叠加平均")
        print("2. ✓ 条件对比图")
        print("3. ✓ 差异波分析 (NoGo - Go)")
        print("4. ✓ 典型电极的ERP波形")
        print("5. ✓ 地形图（如果电极位置正确）")
        print("6. ✓ ERP成分分析 (N200, P300)")
        print("\n下一步建议：")
        print("1. 检查各个图形，确认数据质量")
        print("2. 根据需要调整时间窗口和电极选择")
        print("3. 进行统计分析（例如Go vs NoGo的差异检验）")
        print("4. 导出数据进行进一步分析")

    except Exception as e:
        print(f"\n✗ 预处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    # 显示所有图形
    plt.show()


if __name__ == '__main__':
    main()