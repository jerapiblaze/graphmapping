import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('./debug_q.csv')

# Lấy giá trị của cột 'ep' và 'q'
eps = df['ep']
qvalues = df['q']

# Vẽ đồ thị
plt.plot(eps, qvalues, label='Q Values')
plt.xlabel('Episode')
plt.ylabel('Q Value')
plt.title('Q Values vs Episode')
plt.legend()
plt.show()
