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
            # 如果Curry格式读取失败，尝试其他方法
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

    def remove_problematic_channels(self):
        """移除可能导致可视化问题的电极"""
        print("\n" + "=" * 60)
        print("处理问题电极")
        print("=" * 60)

        # 已知会导致位置重叠的电极
        problematic_channels = ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2']
        channels_to_remove = [ch for ch in problematic_channels if ch in self.raw.ch_names]

        if channels_to_remove:
            print(f"检测到问题电极: {channels_to_remove}")
            response = input("是否移除这些电极以避免可视化错误? (y/n): ")

            if response.lower() == 'y':
                self.raw.drop_channels(channels_to_remove)
                print(f"✓ 已移除电极: {channels_to_remove}")
                print(f"  剩余电极数: {len(self.raw.ch_names)}")
                return True
            else:
                print("⚠ 保留问题电极，但可视化可能会失败")
                return False
        else:
            print("✓ 未检测到问题电极")
            return False

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
                    ch_name=eog_ch_names,  # 明确指定EOG通道
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
            events = np.array([])  # 返回空数组而不是None

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
            事件ID字典，例如 {'Go': 1, 'NoGo': 2}
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

        if event_id is None:
            # 自动创建事件ID
            unique_events = np.unique(events[:, 2])
            event_id = {f'Event_{i}': i for i in unique_events}
            print(f"使用自动生成的事件ID: {event_id}")
            print("⚠ 建议根据实验设计修改事件ID")

        if reject is None:
            reject = {'eeg': 100e-6}  # 100 µV

        print(f"\nEpoch参数:")
        print(f"  时间窗口: {tmin} 到 {tmax} 秒")
        print(f"  基线: {baseline}")
        print(f"  拒绝标准: {reject}")

        self.epochs = mne.Epochs(self.raw, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, baseline=baseline,
                                reject=reject, preload=True, verbose=False)

        print(f"\n✓ 创建了 {len(self.epochs)} 个epochs")
        print(f"  各条件的epoch数量:")
        for event_name, count in zip(self.epochs.event_id.keys(),
                                     [len(self.epochs[ev]) for ev in self.epochs.event_id.keys()]):
            print(f"    {event_name}: {count}")

        # 绘制epochs图像
        self.epochs.plot(n_epochs=5, n_channels=30, scalings='auto')

        return self.epochs

    def plot_erp(self):
        """绘制事件相关电位（ERP）"""
        print("\n" + "=" * 60)
        print("步骤 9: 绘制ERP")
        print("=" * 60)

        if self.epochs is None:
            print("✗ 请先创建epochs")
            return

        # 计算平均ERP
        evoked_dict = {condition: self.epochs[condition].average()
                      for condition in self.epochs.event_id.keys()}

        # 绘制所有条件的ERP对比
        colors = ['blue', 'red', 'green', 'orange']
        try:
            mne.viz.plot_compare_evokeds(evoked_dict, picks='eeg', combine='mean',
                                         colors=colors[:len(evoked_dict)])
            print("✓ ERP波形图绘制完成")
        except Exception as e:
            print(f"⚠ 绘制ERP波形时出现问题: {e}")

        # 绘制地形图
        print("\n尝试绘制地形图...")
        for condition, evoked in evoked_dict.items():
            try:
                times = np.linspace(evoked.tmin, evoked.tmax, 6)[1:-1]
                evoked.plot_topomap(times=times, title=f'{condition} - 地形图',
                                   time_unit='s')
                print(f"✓ {condition} 地形图绘制完成")
            except ValueError as e:
                if "overlapping positions" in str(e):
                    print(f"⚠ {condition} 地形图无法绘制（电极位置重叠）")
                    print("  建议：可以绘制单个电极的ERP波形代替")
                    # 绘制几个典型电极的ERP
                    try:
                        typical_channels = ['Fz', 'Cz', 'Pz']
                        available_channels = [ch for ch in typical_channels if ch in evoked.ch_names]
                        if available_channels:
                            fig, axes = plt.subplots(len(available_channels), 1,
                                                   figsize=(10, 3*len(available_channels)))
                            if len(available_channels) == 1:
                                axes = [axes]
                            for ax, ch in zip(axes, available_channels):
                                evoked.copy().pick_channels([ch]).plot(axes=ax, show=False)
                                ax.set_title(f'{condition} - {ch}')
                            plt.tight_layout()
                            print(f"  ✓ 绘制了 {condition} 在 {available_channels} 电极的波形")
                    except Exception as e2:
                        print(f"  ⚠ 备选可视化也失败: {e2}")
                else:
                    print(f"⚠ {condition} 地形图绘制失败: {e}")
            except Exception as e:
                print(f"⚠ {condition} 地形图绘制失败: {e}")

        print("\n✓ ERP分析完成")

    def save_preprocessed_data(self, output_dir='preprocessed_data'):
        """保存预处理后的数据"""
        print("\n" + "=" * 60)
        print("步骤 10: 保存数据")
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

    # 设置参数
    DATA_PATH = '.'  # 数据文件所在目录
    SUBJECT_NAME = 'Acquisition 190'  # 文件名前缀

    # 创建预处理器
    preprocessor = EEGPreprocessor(DATA_PATH, SUBJECT_NAME)

    try:
        # 1. 加载数据
        raw = preprocessor.load_data()

        # 2. 设置电极位置
        preprocessor.set_montage()

        # 2.5 处理问题电极（可选）
        # 如果你的数据有F11, F12, FT11, FT12, Cb1, Cb2等电极
        # 这些电极可能导致可视化问题
        # preprocessor.remove_problematic_channels()

        # 3. 滤波
        preprocessor.filter_data(l_freq=0.5, h_freq=40)

        # 4. 检测坏导联
        preprocessor.detect_bad_channels()

        # 5. 重参考
        preprocessor.set_reference('average')

        # 6. 运行ICA
        ica = preprocessor.run_ica(n_components=15)

        # 7. 应用ICA（如果需要手动选择成分，请取消下面的注释并指定）
        # preprocessor.apply_ica(exclude_components=[0, 1])  # 示例：排除成分0和1

        # 如果自动检测到了眼电成分，直接应用
        if len(ica.exclude) > 0:
            preprocessor.apply_ica()
        else:
            print("\n⚠ 需要手动选择要排除的ICA成分")
            print("  请查看ICA成分图，然后取消注释上面的apply_ica行并指定成分索引")
            response = input("\n是否继续不应用ICA？(y/n): ")
            if response.lower() != 'y':
                print("程序终止。请检查ICA成分后重新运行。")
                return

        # 8. 查找事件
        events = preprocessor.find_events()

        if len(events) > 0:
            # 9. 创建epochs
            # 注意：请根据你的实验设计修改event_id
            # 例如：event_id = {'Go': 1, 'NoGo': 2}
            event_id = None  # 使用自动检测的事件ID

            epochs = preprocessor.create_epochs(
                events,
                event_id=event_id,
                tmin=-0.2,  # epoch开始时间（秒）
                tmax=0.8,   # epoch结束时间（秒）
                baseline=(None, 0),  # 基线校正
                reject={'eeg': 100e-6}  # 拒绝标准（100 µV）
            )

            # 10. 绘制ERP
            preprocessor.plot_erp()
        else:
            print("\n⚠ 由于没有找到事件标记，跳过epoch创建和ERP分析")

        # 11. 保存数据
        preprocessor.save_preprocessed_data()

        print("\n" + "=" * 60)
        print("预处理完成！")
        print("=" * 60)
        print("\n下一步建议：")
        print("1. 检查ICA成分，确保正确去除了眼电等伪迹")
        print("2. 根据实验设计修改event_id")
        print("3. 调整epoch的时间窗口和基线")
        print("4. 进行进一步的统计分析")

    except Exception as e:
        print(f"\n✗ 预处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    # 显示所有图形
    plt.show()


if __name__ == '__main__':
    main()