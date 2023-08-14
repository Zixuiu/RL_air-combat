# 引入所需的库
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from class_env import AirCombat

# 代码实现了一个飞行作战的环境，并使用深度强化学习算法训练一个网络模型来控制红方飞机的动作。
# 训练过程中，模型根据当前状态预测最优动作，并根据贪婪策略选择动作。训练结束后，使用训练好的模型进行测试，并将结果可视化。

# 定义常量
CHECKPOINT_DIR = './checkpoint/'
EPOCH = 1000
MAX_STEPS = 1000
K = 0.5
alpha = 0.1
gamma = 0.4
EPSILON = 0.1

# 创建网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(72, 100)
        self.layer2 = nn.Linear(100, 30)
        self.layer3 = nn.Linear(30, 9)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x

# 定义贪婪策略
def greedy(prediction, list):
    temp = np.argmax(prediction)
    action = list[temp]
    return action

# 主函数
def main():
    # 创建网络模型对象
    model = Net()
    # 将模型转移到GPU
    model = model.to('cuda')
    # 定义优化器和损失函数
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fun = nn.MSELoss()

    for num in range(EPOCH):
        X = []
        Y = []
        Z = []
        X_B = []
        Y_B = []
        Z_B = []
        print("Episode %d" % num)
        # 创建飞行作战实例
        aircombat = AirCombat()
        # 获取红方和蓝方的起始位置和动作
        position_r_0 = aircombat.position_r 
        action_r_0 = aircombat.active_r
        position_b_0 = aircombat.position_b
        action_b = aircombat.active_b

        # 生成初始的状态 s0
        s0 = aircombat.generate_state(position_r_0, position_b_0, action_r_0, action_b)

        # 获取红方和蓝方的速度和角度
        v_r = action_r_0[0]
        gamma_r = action_r_0[1]
        pusin_r = action_r_0[2]
        
        # 获取红方的动作集合
        action_list = aircombat.action(v_r, gamma_r, pusin_r)
        action_b = aircombat.action_b(action_b)

        # 根据动作集合依次生成新的状态
        for action in action_list:
            position_r_state, position_b_state = aircombat.generate_next_position(position_r_0, position_b_0,
                                                                                action, action_b)
            state = aircombat.generate_state(position_r_state, position_b_state, action, action_b)
            state = aircombat.normalize(state)
            start_input_list.append(state)

        # 格式化输入数据
        start_input_array = np.array(start_input_list)
        start_input = start_input_array.reshape((1, 72))

        # 初始化状态和动作列表
        state_list = []
        state_list.append(start_input)
        
        # 开始游戏
        j = 0
        t_f = True
        while (t_f == True):  
            # 将当前状态设置为网络的输入
            input = torch.unsqueeze(torch.from_numpy(state_now).type(torch.FloatTensor),0).cuda()
            # 前向传播，生成结果
            prediction = model(input)
            prediction = prediction.cpu().detach().numpy()
            # 根据贪婪策略选择动作
            temp = greedy(prediction, action_list)
            action_now = aircombat.action(v_r, gamma_r, pusin_r, flag=False, choose=temp)
            # 生成下一个状态
            position_r_now, position_b_now = aircombat.generate_next_position(position_r_0, position_b_0, action_now, action_b)
            state_next = aircombat.generate_state(position_r_now, position_b_now, action_now, action_b)

            v_r = action_now[0]
            gamma_r = action_now[1]
            pusin_r = action_now[2]
            action_list_now = aircombat.action(v_r, gamma_r, pusin_r, flag=True, choose=1)
            
            # 根据下一个状态生成新的输入数据
            start_input_batch = []
            for action_next in action_list_now:
                position_r_state, position_b_state = aircombat.generate_next_position(position_r_now, position_b_now,
                                                                                    action_next, action_b)
                state_now = aircombat.generate_state(position_r_state, position_b_state, action_next, action_b)
                state_now = aircombat.normalize(state_now)
                start_input_batch.append(state_now)
            state_batch = np.array(start_input_batch)
            state_batch = state_batch.reshape((1, 72))
            state_list.append(state_batch)
            
            position_r_0 = position_r_now
            position_b_0 = position_b_now
            
            x_r = position_r_now[0]
            y_r = position_r_now[1]
            z_r = position_r_now[2]
            x_b = position_b_now[0]
            y_b = position_b_now[1]
            z_b = position_b_now[2]
            
            X.append(x_r)
            Y.append(y_r)
            Z.append(z_r)

            X_B.append(x_b)
            Y_B.append(y_b)
            Z_B.append(z_b)
            
            j += 1
            print('Step %d' % j)
            print('Action:', action_now)
            print('Position (red):', x_r, y_r, z_r)
            print('Position (blue):', x_b, y_b, z_b)
            
            # 判断胜负条件
            q_r = state_next[0]
            q_b = state_next[1]
            d = state_next[2]
            delta_h = state_next[4]
            if j == MAX_STEPS:
                print('Max steps reached')
                t_f = False
                R = -5
            if d < 2500:
                if q_r < 30 and q_b > 30 and delta_h > 0:
                    print('Red wins!')
                    t_f = False
                    R = 10
                if q_r > 30 and q_b < 30 and delta_h < 0:
                    print('Blue wins...')
                    t_f = False
                    R = -10
            if x_r > 200000 or x_r < 0 or y_r > 200000 or y_r < 0 or z_r > 11000 or z_r < 0 or x_b > 200000 or x_b < 0 or y_b > 200000 or y_b < 0 or z_b > 11000 or z_b < 0:
                print('Out of combat area')
                t_f = False
                R = -5

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        X_B = np.array(X_B)
        Y_B = np.array(Y_B)
        Z_B = np.array(Z_B)

        # Save the trained model
        if num % 100 == 0:
            torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/model.pt')

def test():
    model = Net()
    model.state_dict(torch.load(f'{CHECKPOINT_DIR}/model.pt'))
    model = model.to('cuda')
    # 对一切先初始化
    ax1 = plt.axes(projection='3d')
    X = []
    Y = []
    Z = []
    X_B = []
    Y_B = []
    Z_B = []
    aircombat = AirCombat()
    position_r_0 = aircombat.position_r 
    action_r_0 = aircombat.active_r
    position_b_0 = aircombat.position_b
    action_b = aircombat.active_b
    
    s0 = aircombat.generate_state(position_r_0, position_b_0, action_r_0, action_b)

    v_r = action_r_0[0]
    gamma_r = action_r_0[1]
    pusin_r = action_r_0[2]
    action_list = aircombat.action(v_r, gamma_r, pusin_r)
    action_b = aircombat.action_b(action_b)

    for action in action_list:
        position_r_state, position_b_state = aircombat.generate_next_position(position_r_0, position_b_0,
                                                                                action, action_b)
        state = aircombat.generate_state(position_r_state, position_b_state, action, action_b)
        state = aircombat.normalize(state)
        start_input_list.append(state)

    start_input_array = np.array(start_input_list)
    start_input = start_input_array.reshape((1, 72))
    state_list = []
    state_list.append(start_input)
    
    j = 0
    t_f = True
    while (t_f == True):  
        input = torch.unsqueeze(torch.from_numpy(state_now).type(torch.FloatTensor), 0).cuda()
        prediction = model(input)
        prediction = prediction.cpu().detach().numpy()
        temp = np.argmax(prediction)
        action_now = aircombat.action(v_r, gamma_r, pusin_r, flag=False, choose=temp)
        position_r_now, position_b_now = aircombat.generate_next_position(position_r_0, position_b_0, action_now, action_b)

        state_next = aircombat.generate_state(position_r_now, position_b_now, action_now, action_b)

        v_r = action_now[0]
        gamma_r = action_now[1]
        pusin_r = action_now[2]
        action_list_now = aircombat.action(v_r, gamma_r, pusin_r, flag=True, choose=1)

        start_input_batch = []
        for action_next in action_list_now:
            position_r_state, position_b_state = aircombat.generate_next_position(position_r_now, position_b_now,
                                                                                    action_next, action_b)
            state_now = aircombat.generate_state(position_r_state, position_b_state, action_next, action_b)
            state_now = aircombat.normalize(state_now)
            start_input_batch.append(state_now)
        state_batch = np.array(start_input_batch)
        state_batch = state_batch.reshape((1, 72))
        state_list.append(state_batch)

        position_r_0 = position_r_now
        position_b_0 = position_b_now

        x_r = position_r_now[0]
        y_r = position_r_now[1]
        z_r = position_r_now[2]
        x_b = position_b_now[0]
        y_b = position_b_now[1]
        z_b = position_b_now[2]

        X.append(x_r)
        Y.append(y_r)
        Z.append(z_r)

        X_B.append(x_b)
        Y_B.append(y_b)
        Z_B.append(z_b)

        j += 1
        print('Step %d' % j)
        print('Action:', action_now)
        print('Position (red):', x_r, y_r, z_r)
        print('Position (blue):', x_b, y_b, z_b)

        q_r = state_next[0]
        q_b = state_next[1]
        d = state_next[2]
        delta_h = state_next[4]
        if j == MAX_STEPS:
            print('Max steps reached')
            t_f = False
            R = -5
        if d < 2500:
            if q_r < 30 and q_b > 30 and delta_h > 0:
                print('Red wins!')
                t_f = False
                R = 10
            if q_r > 30 and q_b < 30 and delta_h < 0:
                print('Blue wins...')
                t_f = False
                R = -10
        if x_r > 200000 or x_r < 0 or y_r > 200000 or y_r < 0 or z_r > 11000 or z_r < 0 or x_b > 200000 or x_b < 0 or y_b > 200000 or y_b < 0 or z_b > 11000 or z_b < 0:
            print('Out of combat area')
            t_f = False
            R = -5
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    X_B = np.array(X_B)
    Y_B = np.array(Y_B)
    Z_B = np.array(Z_B)

    # 绘制场景
    ax1.scatter3D(X_B, Y_B, Z_B, label='Blue')
    ax1.plot3D(X, Y, Z, 'gray')
    ax1.scatter3D(X, Y, Z, label='Red')
    plt.legend()

    if R==10:
        flag='Red'
    elif R==-10:
        flag='Blue'
    else:
        flag='error'
    plt.savefig(f'{flag}.svg')
    plt.show()
    

if __name__ == '__main__':
    # 运行训练
    main()
    # 运行测试
    test()
