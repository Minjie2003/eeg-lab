import mne
import numpy as np
import os
import matplotlib

# 使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.preprocessing import ICA


def process_cdt_64_separate(cdt_file_path, output_dir='processed_data_separate'):
    """
    CDT文件处理流程 - 生成64个电极的单独曲线图
    """
    print("=" * 70)
    print("CDT File Processing Pipeline - 64 Separate Channels")
    print("=" * 70)

    # 创建目录
    figures_dir = os.path.join(output_dir, 'figures')
    channels_dir = os.path.join(figures_dir, 'individual_channels')
    if not os.path.exists(channels_dir):
        os.makedirs(channels_dir)

    # 步骤1: 加载CDT文件
    print("\n1. LOADING CDT FILE")
    try:
        raw = mne.io.read_raw(cdt_file_path, preload=True)
        print(f"  ✓ Data loaded successfully")
        print(f"     Channels: {len(raw.ch_names)}")
        print(f"     Sampling rate: {raw.info['sfreq']} Hz")
        print(f"     Duration: {raw.times[-1]:.2f} seconds")

    except Exception as e:
        print(f"  ❌ Failed to load CDT file: {e}")
        return None, None, None

    # 检查事件标记
    print("\n🔍 CHECKING EVENT MARKERS")
    annotations = raw.annotations
    print(f"  Found {len(annotations)} event markers")

    if len(annotations) > 0:
        unique_events = set(annotations.description)
        print(f"  Event types: {unique_events}")

        for event_type in unique_events:
            count = np.sum(annotations.description == event_type)
            print(f"    {event_type}: {count} times")
    else:
        print("  ❌ No event markers found")
        return raw, None, None

    # 步骤2: 电极定位
    print("\n2. ELECTRODE POSITIONING")
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        print("  ✓ Standard 10-20 montage applied")

    except Exception as e:
        print(f"  ! Electrode positioning warning: {e}")

    # 步骤3: 剔除无用电极
    print("\n3. REMOVING UNNECESSARY CHANNELS")
    eeg_channels = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, ecg=False)
    original_ch_count = len(raw.ch_names)
    raw.pick([raw.ch_names[i] for i in eeg_channels])
    print(f"  ✓ Kept {len(raw.ch_names)} EEG channels from {original_ch_count} total channels")

    # 步骤4: 重参考
    print("\n4. RE-REFERENCING")
    raw.set_eeg_reference(ref_channels='average')
    print("  ✓ Average reference applied")

    # 步骤5: 滤波
    print("\n5. FILTERING")
    print("  Applying bandpass filter (0.1-40Hz)...")
    raw.filter(0.1, 40.0, fir_design='firwin')
    print("  ✓ Bandpass filter completed")

    print("  Applying notch filter (50Hz noise)...")
    raw.notch_filter(np.arange(50, 251, 50))
    print("  ✓ Notch filter completed")

    # 步骤6: 分段
    print("\n6. EPOCHING")

    events, event_id = mne.events_from_annotations(raw)
    print(f"  ✓ Created {len(events)} events")
    print(f"  Event ID mapping: {event_id}")

    # 显示每个事件类型的trials数量
    print("\n  Trials per event type:")
    for event_name, event_num in event_id.items():
        count = np.sum(events[:, 2] == event_num)
        print(f"    {event_name}: {count} trials")

    # 创建epochs
    tmin = -0.2
    tmax = 1.0

    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        baseline=(tmin, 0),
                        preload=True)

    print(f"  ✓ Successfully created {len(epochs)} epochs")

    # 保存分段数据
    epochs_save_path = os.path.join(output_dir, 'step6_epochs-epo.fif')
    epochs.save(epochs_save_path, overwrite=True)
    print(f"  💾 Epochs data saved: {epochs_save_path}")

    # 步骤7-8: ICA处理
    print("\n7-8. ICA PROCESSING AND ARTIFACT REMOVAL")

    # 为ICA准备数据
    raw_ica = raw.copy()
    raw_ica.filter(1., None)

    # 运行ICA
    ica = ICA(n_components=15, random_state=97, max_iter=800)
    print("  Fitting ICA...")
    ica.fit(raw_ica)
    print("  ✓ ICA fitting completed")

    # 自动检测眼电
    print("  Detecting EOG components...")
    eog_indices, eog_scores = ica.find_bads_eog(raw_ica, ch_name=['Fp1', 'Fp2', 'Fpz'])

    if eog_indices:
        print(f"  Detected EOG components: {eog_indices}")
        ica.exclude = eog_indices

        # 应用ICA清理
        print("  Applying ICA cleaning...")
        ica.apply(epochs)
        print("  ✓ ICA artifact removal completed")

        # 重新进行基线校正
        epochs.apply_baseline(baseline=(tmin, 0))
        print("  ✓ Baseline correction reapplied")
    else:
        print("  No significant EOG components detected automatically")

    # 保存ICA数据
    ica_save_path = os.path.join(output_dir, 'ica_result-ica.fif')
    ica.save(ica_save_path, overwrite=True)
    print(f"  💾 ICA data saved: {ica_save_path}")

    # 步骤9: 保存最终数据和生成64电极单独曲线图
    print("\n9. SAVING FINAL RESULTS AND GENERATING 64 SEPARATE CHANNEL PLOTS")

    # 保存最终数据
    final_epochs_path = os.path.join(output_dir, 'final_cleaned_epochs-epo.fif')
    epochs.save(final_epochs_path, overwrite=True)
    print(f"  💾 Final cleaned epochs saved: {final_epochs_path}")

    # 生成64个电极的单独曲线图
    generate_64_separate_channel_plots(epochs, event_id, channels_dir)

    # 创建处理报告
    create_separate_report(raw, epochs, event_id, output_dir, channels_dir)

    return raw, epochs, ica


def generate_64_separate_channel_plots(epochs, event_id, channels_dir):
    """
    为每个电极生成单独的曲线图
    """
    print("  Generating 64 separate channel plots...")

    # 对每个事件类型生成单独的电极图
    for event_name in event_id.keys():
        try:
            # 检查该事件类型是否有trials
            if event_name in epochs.event_id and len(epochs[event_name]) > 0:
                print(f"    Processing event: {event_name}")

                # 为该事件类型创建单独的目录
                event_dir = os.path.join(channels_dir, f'event_{event_name}')
                if not os.path.exists(event_dir):
                    os.makedirs(event_dir)

                # 计算该事件类型的平均ERP
                evoked = epochs[event_name].average()

                # 为每个通道生成单独的图
                for channel_name in evoked.ch_names:
                    try:
                        # 创建单个通道的图
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # 获取该通道的数据
                        channel_idx = evoked.ch_names.index(channel_name)
                        channel_data = evoked.data[channel_idx]
                        times = evoked.times

                        # 绘制单个通道的ERP
                        ax.plot(times, channel_data, linewidth=2, color='blue')

                        # 添加标记线
                        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

                        # 设置图表属性
                        ax.set_title(f'Channel {channel_name} - Event {event_name}', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Time (s)', fontsize=12)
                        ax.set_ylabel('Amplitude (µV)', fontsize=12)
                        ax.grid(True, alpha=0.3)
                        ax.legend()

                        # 设置坐标轴范围
                        ax.set_xlim(times[0], times[-1])

                        # 保存单个通道的图
                        channel_filename = f'channel_{channel_name}_event_{event_name}.png'
                        channel_path = os.path.join(event_dir, channel_filename)
                        plt.savefig(channel_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)

                    except Exception as e:
                        print(f"      ! Failed to save channel {channel_name}: {e}")

                print(f"      💾 Generated {len(evoked.ch_names)} individual channel plots for event {event_name}")

                # 同时生成按脑区分组的汇总图
                generate_brain_region_summary(evoked, event_name, event_dir)

        except Exception as e:
            print(f"    ! Failed to generate plots for event {event_name}: {e}")

    print("  ✓ All 64 separate channel plots generated")


def generate_brain_region_summary(evoked, event_name, event_dir):
    """
    生成按脑区分组的汇总图（可选）
    """
    try:
        # 定义脑区
        brain_regions = {
            'Frontal': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'AF3', 'AF4', 'AF7', 'AF8'],
            'Central': ['Cz', 'C3', 'C4', 'C1', 'C2', 'C5', 'C6', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
            'Parietal': ['Pz', 'P3', 'P4', 'P1', 'P2', 'P5', 'P6', 'P7', 'P8', 'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5',
                         'CP6'],
            'Temporal': ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
            'Occipital': ['Oz', 'O1', 'O2', 'POz', 'PO3', 'PO4', 'PO7', 'PO8']
        }

        # 为每个脑区生成汇总图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (region_name, region_channels) in enumerate(brain_regions.items()):
            if i < len(axes):
                # 找到该脑区存在的通道
                available_channels = [ch for ch in region_channels if ch in evoked.ch_names]

                if available_channels:
                    # 绘制该脑区所有通道的曲线
                    for channel in available_channels:
                        channel_idx = evoked.ch_names.index(channel)
                        channel_data = evoked.data[channel_idx]
                        axes[i].plot(evoked.times, channel_data, label=channel, alpha=0.7, linewidth=1)

                    axes[i].set_title(f'{region_name} Region', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('Time (s)')
                    axes[i].set_ylabel('Amplitude (µV)')
                    axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                    axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend(fontsize=8)

        # 隐藏多余的子图
        for i in range(len(brain_regions), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Brain Region Summary - Event {event_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(event_dir, f'brain_region_summary_{event_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"      💾 Brain region summary saved for event {event_name}")

    except Exception as e:
        print(f"      ! Failed to generate brain region summary: {e}")


def create_separate_report(raw, epochs, event_id, output_dir, channels_dir):
    """
    创建处理报告
    """
    report_path = os.path.join(output_dir, 'processing_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CDT File Processing Report - 64 Separate Channels\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATA INFORMATION:\n")
        f.write(f"- Original channels: {len(raw.ch_names)}\n")
        f.write(f"- Sampling rate: {raw.info['sfreq']} Hz\n")
        f.write(f"- Data duration: {raw.times[-1]:.2f} seconds\n\n")

        f.write("PROCESSING RESULTS:\n")
        f.write(f"- Final trials count: {len(epochs)}\n")
        f.write(f"- Event types: {list(event_id.keys())}\n\n")

        f.write("TRIALS PER EVENT TYPE:\n")
        for event_name in event_id.keys():
            if event_name in epochs.event_id:
                count = len(epochs[event_name])
                f.write(f"  - {event_name}: {count} trials\n")

        f.write(f"\nGENERATED FILES:\n")
        # 统计生成的图片数量
        total_images = 0
        if os.path.exists(channels_dir):
            for root, dirs, files in os.walk(channels_dir):
                png_files = [f for f in files if f.endswith('.png')]
                total_images += len(png_files)
                rel_path = os.path.relpath(root, channels_dir)
                if png_files:
                    f.write(f"  - {rel_path}/: {len(png_files)} images\n")

        f.write(f"\nTOTAL IMAGES GENERATED: {total_images}\n")
        f.write(f"OUTPUT DIRECTORY: {output_dir}\n")

    print(f"  💾 Processing report saved: {report_path}")
    print(f"  📊 Total images generated: {total_images}")


# 主程序
if __name__ == "__main__":
    cdt_file_path = "Acquisition 190.cdt"
    output_dir = "processed_data_separate_channels"

    print("🚀 STARTING CDT FILE PROCESSING - 64 SEPARATE CHANNELS")
    print(f"Input file: {cdt_file_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    try:
        # 运行处理流程
        raw_processed, epochs_processed, ica_obj = process_cdt_64_separate(
            cdt_file_path, output_dir
        )

        if epochs_processed is not None:
            print("\n" + "=" * 50)
            print("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"✓ Created {len(epochs_processed)} trials")
            print(f"✓ Generated individual plots for all 64 channels")
            print(f"✓ All results saved to: {output_dir}")
            print(f"✓ Check processing_report.txt for details")

        else:
            print("\n⚠️ Processing completed but no epochs were created")

    except Exception as e:
        print(f"\n❌ PROCESSING FAILED: {e}")
        import traceback

        traceback.print_exc()