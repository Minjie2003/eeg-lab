"""
Go/NoGo 脑电数据预处理脚本 - 自动模式
此版本会自动处理电极位置重叠问题
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
import warnings

warnings.filterwarnings('ignore')

# 从main.py导入预处理器类
from main import EEGPreprocessor, check_dependencies

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """主函数 - 自动模式"""
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Go/NoGo EEG 数据预处理 (自动模式)" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # 检查依赖包
    if not check_dependencies():
        sys.exit(1)

    # 设置参数
    DATA_PATH = '.'
    SUBJECT_NAME = 'Acquisition 190'

    # 创建预处理器
    preprocessor = EEGPreprocessor(DATA_PATH, SUBJECT_NAME)

    try:
        # 1. 加载数据
        raw = preprocessor.load_data()

        # 2. 自动移除问题电极
        problematic_channels = ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2']
        channels_to_remove = [ch for ch in problematic_channels if ch in raw.ch_names]

        if channels_to_remove:
            print("\n" + "=" * 60)
            print("自动处理问题电极")
            print("=" * 60)
            print(f"检测到问题电极: {channels_to_remove}")
            print("这些电极会导致可视化错误，将自动移除")
            raw.drop_channels(channels_to_remove)
            print(f"✓ 已移除电极: {channels_to_remove}")
            print(f"  剩余电极数: {len(raw.ch_names)}")

        # 3. 设置电极位置
        preprocessor.set_montage()

        # 4. 滤波
        preprocessor.filter_data(l_freq=0.5, h_freq=40)

        # 5. 检测坏导联
        preprocessor.detect_bad_channels()

        # 6. 重参考
        preprocessor.set_reference('average')

        # 7. 运行ICA
        print("\n" + "=" * 60)
        print("运行ICA（这可能需要几分钟）...")
        print("=" * 60)
        ica = preprocessor.run_ica(n_components=15)

        # 8. 应用ICA
        if len(ica.exclude) > 0:
            preprocessor.apply_ica()
        else:
            print("\n⚠ 未自动检测到眼电成分")
            print("如果需要手动选择ICA成分，请查看ICA可视化图")
            print("然后在代码中指定: preprocessor.apply_ica(exclude_components=[0, 1, ...])")

        # 9. 查找事件
        events = preprocessor.find_events()

        if len(events) > 0:
            # 10. 创建epochs
            print("\n建议：根据你的实验设计修改event_id")
            print("例如: event_id = {'Go': 1, 'NoGo': 2}")

            epochs = preprocessor.create_epochs(
                events,
                event_id=None,  # 使用自动检测
                tmin=-0.2,
                tmax=0.8,
                baseline=(None, 0),
                reject={'eeg': 100e-6}
            )

            # 11. 绘制ERP
            preprocessor.plot_erp()
        else:
            print("\n⚠ 未找到事件标记，跳过epoch创建")

        # 12. 保存数据
        preprocessor.save_preprocessed_data()

        print("\n" + "=" * 60)
        print("✓ 预处理完成！")
        print("=" * 60)
        print("\n处理总结:")
        print(f"  - 原始数据: {SUBJECT_NAME}")
        print(f"  - 电极数: {len(preprocessor.raw.ch_names)}")
        if channels_to_remove:
            print(f"  - 已移除电极: {channels_to_remove}")
        if hasattr(preprocessor, 'epochs') and preprocessor.epochs is not None:
            print(f"  - Epochs总数: {len(preprocessor.epochs)}")
        print(f"  - 数据保存在: preprocessed_data/")

        print("\n下一步建议:")
        print("1. 检查ICA成分，确认伪迹去除效果")
        print("2. 根据实验设计修改event_id")
        print("3. 进行统计分析和时频分析")

    except Exception as e:
        print(f"\n✗ 预处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    plt.show()


if __name__ == '__main__':
    main()