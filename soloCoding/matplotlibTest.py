import matplotlib.pyplot as plt

data_dict = {'data_x': [], 'data_y': []}

for x in range(1, 11):
    data_dict['data_x'].append(x)
    data_dict['data_y'].append(10*x)

plt.plot('data_x', 'data_y', data=data_dict)
plt.show()