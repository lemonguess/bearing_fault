
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
# 生成三个不同频率成分的信号
fs = 1000  # 采样率
t = np.linspace(0, 1, fs, endpoint=False)  # 时间

# 第一个频率成分
signal1 = np.sin(2 * np.pi * 50 * t)
# 第二个频率成分
signal2 = np.sin(2 * np.pi * 120 * t)
# 第三个频率成分
signal3 = np.sin(2 * np.pi * 300 * t)

# 合并三个信号
signal = np.concatenate((signal1, signal2, signal3))

# 进行短时傅里叶变换  
f, t, spectrum = stft(signal, fs, nperseg=100, noverlap=50)
# 绘制时频图
plt.pcolormesh(t, f, np.abs(spectrum), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()