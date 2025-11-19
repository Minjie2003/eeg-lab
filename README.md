# Go/NoGo 脑电数据预处理脚本

这是一个用于处理Go/NoGo实验脑电数据的Python预处理脚本，使用MNE-Python库。

## 安装依赖

```bash
pip install -r requirements.txt
```

或者单独安装：
```bash
pip install numpy matplotlib mne scipy
```

## 使用方法

### 1. 准备数据

将你的Curry格式数据文件（.cdt, .cdt.ceo, .cdt.dpa）放在与main.py相同的目录下。

### 2. 运行脚本

```bash
python main.py
```

### 3. 查看结果

脚本会自动：
- 加载数据
- 进行滤波（0.5-40 Hz）
- 检测坏导联
- 重参考（平均参考）
- 运行ICA去除眼电等伪迹
- 查找事件标记
- 创建epochs
- 绘制ERP
- 保存预处理后的数据

## 预处理步骤说明

### 1. 数据加载
- 读取Curry格式的脑电数据
- 显示基本信息（采样率、通道数、数据长度）

### 2. 电极位置设置
- 使用标准10-20电极系统

### 3. 滤波
- 默认：0.5-40 Hz 带通滤波
- 可根据需要调整频率范围

### 4. 坏导联检测
- 自动检测异常通道
- 基于标准差的方法

### 5. 重参考
- 默认使用平均参考
- 可选择特定参考电极

### 6. ICA伪迹去除
- 运行独立成分分析
- 自动检测眼电成分（如果有EOG通道）
- 可视化ICA成分供检查

### 7. 事件标记
- 从STIM通道自动查找事件
- 显示事件类型和数量

### 8. Epochs创建
- 时间窗口：-0.2 到 0.8 秒（可调整）
- 基线校正：刺激前200ms
- 拒绝标准：100 µV（可调整）

### 9. ERP分析
- 计算各条件的平均波形
- 绘制ERP对比图
- 绘制地形图

### 10. 数据保存
- 保存在 `preprocessed_data/` 目录
- 包括：预处理后的raw数据、epochs、ICA

## 自定义参数

在`main()`函数中可以修改以下参数：

```python
# 文件路径
DATA_PATH = '.'
SUBJECT_NAME = 'Acquisition 190'

# 滤波参数
preprocessor.filter_data(l_freq=0.5, h_freq=40)

# ICA参数
preprocessor.run_ica(n_components=15)

# Epoch参数
preprocessor.create_epochs(
    events, 
    event_id={'Go': 1, 'NoGo': 2},  # 根据你的实验修改
    tmin=-0.2,
    tmax=0.8,
    baseline=(None, 0),
    reject={'eeg': 100e-6}
)
```

## Go/NoGo实验说明

Go/NoGo范式是研究执行功能和抑制控制的经典实验：

- **Go试次**：需要快速做出反应
- **NoGo试次**：需要抑制反应

### 典型的ERP成分

1. **N2（200-300ms）**
   - NoGo试次中更明显
   - 反映冲突监测和抑制控制

2. **P3（300-500ms）**
   - Go和NoGo试次都有
   - 反映注意和认知评估

3. **NoGo-P3**
   - NoGo试次特有的P3成分
   - 反映抑制过程

## 常见问题

### Q: 如果没有找到事件标记怎么办？

A: 可能需要：
1. 检查STIM通道名称
2. 从annotations中提取事件
3. 手动创建事件数组

### Q: 如何手动选择ICA成分？

A: 
1. 查看ICA成分图
2. 识别眼电、心电等伪迹成分
3. 在代码中指定：`preprocessor.apply_ica(exclude_components=[0, 1, 2])`

### Q: 如何修改event_id？

A: 根据你的实验设计，在`create_epochs()`中设置：
```python
event_id = {
    'Go': 1,      # Go试次的事件代码
    'NoGo': 2     # NoGo试次的事件代码
}
```

## 输出文件

预处理后的数据保存在 `preprocessed_data/` 目录：

- `Acquisition 190_raw_preprocessed-raw.fif` - 预处理后的连续数据
- `Acquisition 190_epochs-epo.fif` - 分段数据
- `Acquisition 190_ica.fif` - ICA对象

这些文件可以用于后续分析，例如：
```python
import mne

# 加载epochs
epochs = mne.read_epochs('preprocessed_data/Acquisition 190_epochs-epo.fif')

# 进行统计分析
# ...
```

## 后续分析建议

1. **时频分析**：研究不同频段的功率变化
2. **统计分析**：比较Go和NoGo条件的差异
3. **源定位**：使用MNE进行源空间分析
4. **机器学习**：训练分类器区分不同条件

## 技术支持

如有问题，可以：
- 查看MNE-Python文档：https://mne.tools
- 检查数据格式是否正确
- 确保所有依赖包已安装

## 许可

本脚本仅供学习和研究使用。
