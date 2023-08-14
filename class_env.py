import numpy as np
import random
import math
import random
'''这一版不考虑速度的改变'''
import numpy

# 代码实现了一个基于ε-greedy策略的动作选择函数。
# 给定一个动作值函数的预测结果prediction和一个探索率epsilon，函数会以概率1-epsilon选择预测结果中具有最大值的动作，以概率epsilon随机选择一个动作。
# 其中，np.argmax(prediction)用于找到预测结果中最大值的索引，random.randint(0,8)用于随机选择一个0到8之间的整数作为动作的索引。最后，函数返回选择的动作索引。

#创建环境类
class AirCombat():
    def __init__(self):
        #初始化状态空间
        self.position_r = [130000,100000,3000]  # 我方飞机的初始位置
        self.position_b = [130000,110000,3000]  # 敌方飞机的初始位置
        #初始化动作空间
    #执行一步动作，转换一次状态，返回回报值和下一个状态
        x_r = self.position_r[0]
        x_b = self.position_b[0]
        y_r = self.position_r[1]
        y_b = self.position_b[1]
        z_r = self.position_r[2]
        z_b = self.position_b[2]
        self.gamma_r = 0
        self.gamma_b = 0

        self.pusin_r = 0
        self.pusin_b = 180

        self.v_r = 250
        self.v_b = 250
        self.active_r = [self.v_r,self.gamma_r,self.pusin_r]
        self.active_b = [self.v_b,self.gamma_b,self.pusin_b]


    '''输入当前位置状态信息和选择后的动作信息，得到下一步的位置状态信息'''
    def generate_next_position(self,position_r,position_b,action_now,action_b):
        x_r = position_r[0]
        x_b = position_b[0]
        y_r = position_r[1]
        y_b = position_b[1]
        z_r = position_r[2]
        z_b = position_b[2]

        v_r = action_now[0]
        gamma_r = action_now[1]
        pusin_r = action_now[2]

        v_b= action_b[0]
        gamma_b= action_b[1]
        pusin_b= action_b[2]

        x_r_ = v_r*math.cos(gamma_r*(3.141592653/180))*math.sin(pusin_r*(3.141592653/180))  # 飞机 x 方向上的变化
        y_r_ = v_r*math.cos(gamma_r*(3.141592653/180))*math.cos(pusin_r*(3.141592653/180))  # 飞机 y 方向上的变化
        z_r_ = v_r*math.sin(gamma_r*(3.141592653/180))  # 飞机 z 方向上的变化

        x_b_ = v_b*math.cos(gamma_b*(3.141592653/180))*math.sin(pusin_b*(3.141592653/180))  # 下一步敌方飞机 x 方向上的变化
        y_b_ = v_b*math.cos(gamma_b*(3.141592653/180))*math.cos(pusin_b*(3.141592653/180))  # 下一步敌方飞机 y 方向上的变化
        z_b_ = v_b*math.sin(gamma_b*(3.141592653/180))  # 下一步敌方飞机 z 方向上的变化

        x_r_next = x_r + x_r_  # 下一步我方飞机在 x 方向上的位置
        y_r_next = y_r + y_r_  # 下一步我方飞机在 y 方向上的位置
        z_r_next = z_r + z_r_  # 下一步我方飞机在 z 方向上的位置

        x_b_next = x_b + x_b_  # 下一步敌方飞机在 x 方向上的位置
        y_b_next = y_b + y_b_  # 下一步敌方飞机在 y 方向上的位置        z_b_next = z_b + z_b_  # 下一步敌方飞机在 z 方向上的位置

        position_r_next = [x_r_next,y_r_next,z_r_next]  # 下一步我方飞机的位置
        position_b_next = [x_b_next,y_b_next,z_b_next]  # 下一步敌方飞机的位置

        return position_r_next,position_b_next

    '''输入当前的位置信息和速度角度等状态信息，得到其所对应的态势信息'''
    def generate_state(self,position_r,position_b,action_r,action_b):
        x_r = position_r[0]
        x_b = position_b[0]
        y_r = position_r[1]
        y_b = position_b[1]
        z_r = position_r[2]
        z_b = position_b[2]
        v_r= action_r[0]
        gamma_r= action_r[1]
        pusin_r= action_r[2]
        v_b= action_b[0]
        gamma_b= action_b[1]
        pusin_b= action_b[2]
        d = math.sqrt((x_r-x_b)**2+(y_r-y_b)**2+(z_r-z_b)**2)  # 我方飞机和敌方飞机之间的距离
        q_r = math.acos(((x_b-x_r)*math.cos(gamma_r*(3.141592653/180))*math.sin(pusin_r*(3.141592653/180))+(y_b-y_r)*math.cos(gamma_r*(3.141592653/180))*math.cos(pusin_r*(3.141592653/180))+(z_b-z_r)*math.sin(gamma_r*(3.141592653/180)))/d)  # 我方飞机和敌方飞机之间的尾舱角
        q_r_ = q_r*(180/3.141592653)  # 将尾舱角转换为角度
        q_b = math.acos(((x_r-x_b)*math.cos(gamma_b)*math.sin(pusin_b)+(y_r-y_b)*math.cos(gamma_b)*math.cos(pusin_b)+(z_r-z_b)*math.sin(gamma_b))/d)  # 敌方飞机和我方飞机之间的尾舱角
        q_b_ = q_b*(180/3.141592653)  # 将尾舱角转换为角度
        beta = math.acos(math.cos(gamma_r*(3.141592653/180))*math.sin(pusin_r*(3.141592653/180))*math.cos(gamma_b*(3.141592653/180))*math.sin(pusin_b*(3.141592653/180))+math.cos(gamma_r*(3.141592653/180))*math.cos(pusin_r*(3.141592653/180))*math.cos(gamma_b*(3.141592653/180))*math.cos(pusin_b*(3.141592653/180))+math.sin(gamma_r*(3.141592653/180))*math.sin(gamma_b*(3.141592653/180)))  # 拦截角
        beta_ = beta*(180/3.141592653)  # 将拦截角转换为角度
        delta_h = z_r-z_b  # 垂直高度差
        delta_v2 = v_r**2-v_b**2  # 速度差的平方
        v2 = v_r**2  # 我方飞机的速度平方
        h = z_r  # 我方飞机的高度
        taishi = [q_r_,q_b_,d,beta_,delta_h,delta_v2,v2,h]  # 返回得到的态势信息
        return taishi

    '''注意角度问题,尚未做出更改'''
    '''输入当前的状态以及对动作的选择，得到当前的状态对应动作的下一步状态'''
    def action(self,v_r,gamma_r,pusin_r,flag,choose):

        #向右为正方向
        gamma_r_increase = gamma_r + 10  # 将旋转角度增加10度
        gamma_r_decrease = gamma_r - 10  # 将旋转角度减小10度
        gamma_r_constant = gamma_r  # 保持当前旋转角度不变
        pusin_r_increase = pusin_r + 10  # 将俯仰角增加10度
        pusin_r_decrease = pusin_r - 10  # 将俯仰角减小10度
        pusin_r_constant = pusin_r  # 保持当前俯仰角不变

        action_1 = [v_r,gamma_r_increase,pusin_r_increase]  # 选择 1 号动作：速度不变，旋转角增加，俯仰角增加
        action_2 = [v_r,gamma_r_increase,pusin_r_constant]  # 选择 2 号动作：速度不变，旋转角增加，俯仰角不变
        action_3 = [v_r,gamma_r_increase,pusin_r_decrease]  # 选择 3 号动作：速度不变，旋转角增加，俯仰角减小
        action_4 = [v_r,gamma_r_constant,pusin_r_increase]  # 选择 4 号动作：速度不变，旋转角不变，俯仰角增加
        action_5 = [v_r,gamma_r_constant,pusin_r_constant]  # 选择 5 号动作：速度不变，旋转角不变，俯仰角不变
        action_6 = [v_r,gamma_r_constant,pusin_r_decrease]  # 选择 6 号动作：速度不变，旋转角不变，俯仰角减小
        action_7 = [v_r,gamma_r_decrease,pusin_r_increase]  # 选择 7 号动作：速度不变，旋转角减小，俯仰角增加
        action_8 = [v_r,gamma_r_decrease,pusin_r_constant]  # 选择 8 号动作：速度不变，旋转角减小，俯仰角不变
        action_9 = [v_r,gamma_r_decrease,pusin_r_decrease]  # 选择 9 号动作：速度不变，旋转角减小，俯仰角减小
        if flag == True:
            return [action_1,action_2,action_3,action_4,action_5,action_6,action_7,action_8,action_9]
        else:
            if choose == 0:
                return action_1
            elif choose == 1:
                return action_2
            elif choose == 2:
                return action_3
            elif choose == 3:
                return action_4
            elif choose == 4:
                return action_5
            elif choose == 5:
                return action_6
            elif choose == 6:
                return action_7
            elif choose == 7:
                return action_8
            elif choose == 8:
                return action_9

    def action_b(self,action_b):
        v_b_ = action_b[0]
        gamma_b = action_b[1]
        pusin_b = action_b[2]
        action_b = [v_b_,gamma_b,pusin_b]
        return action_b

    def normalize(self,array):
        array[0] = array[0] / 200  # 归一化第一个维度（在[0, 200]范围内）
        array[1] = array[1] / 200  # 归一化第二个维度（在[0, 200]范围内）
        array[2] = array[2] / 20000  # 归一化第三个维度（在[0, 20000]范围内）
        array[3] = array[3] / 200  # 归一化第四个维度（在[0, 200]范围内）
        array[4] = array[4] / 10000  # 归一化第五个维度（在[0, 10000]范围内）
        array[5] = array[5]           # 第六个维度无需归一化，维持原样
        array[6] = array[6] / 40000  # 归一化第七个维度（在[0, 40000]范围内）
        array[7] = array[7] / 10000  # 归一化第八个维度（在[0, 10000]范围内）
        return array
    #def normal(self,state_array):
    def position_clip(self,position_r_list,position_b_list):
        x_r = position_r_list[0]
        x_b = position_b_list[0]
        if x_r>(x_b+100) or x_r<(x_b-100):
            x_r = x_b+random.randint(0,10)  # 如果我方飞机和敌方飞机的 x 坐标相差大于100，则将我方飞机的 x 坐标设为敌方飞机的 x 坐标加上0到10之间的随机整数
        position_r_list[0] = x_r
        return position_r_list
    def epsilon_greedy(self,prediction,epsilon):
        num = random.random()
        if num>epsilon:
            temp = np.argmax(prediction)  # 以大概率选择最优动作
        else:
            temp = random.randint(0,8)  # 以小概率随机选择动作
        return temp
