#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Human3.6M 数据集的 txt 文件中提取特定动作
"""

def filter_action(input_file, output_file, action_name='Walking'):
    """
    从输入文件中筛选特定动作，保存到输出文件
    
    Args:
        input_file: 输入txt文件路径
        output_file: 输出txt文件路径
        action_name: 要筛选的动作名称
    """
    filtered_lines = []
    total_lines = 0
    matched_lines = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            total_lines += 1
            
            # 从路径中提取动作名称
            # 例如：images/S1_Directions.54138969_ -> Directions
            parts = line.split(',')
            if len(parts) != 3:
                continue
            
            path = parts[0]
            filename = path.split('/')[-1]  # 获取文件名部分
            # 提取动作名：S1_Walking.xxx -> Walking
            action = filename.split('.')[0].split('_')[-1]
            
            # 如果匹配目标动作，保存这行
            if action == action_name:
                filtered_lines.append(line)
                matched_lines += 1
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        for line in filtered_lines:
            f.write(line + '\n')
    
    return total_lines, matched_lines

if __name__ == "__main__":
    action_to_filter = 'Walking'
    
    print(f"开始提取动作: {action_to_filter}\n")
    
    # 处理 train.txt
    print("处理 train.txt...")
    train_total, train_matched = filter_action(
        'images\\train.txt', 
        'train_walking.txt', 
        action_to_filter
    )
    print(f"  总行数: {train_total}")
    print(f"  匹配行数: {train_matched}")
    print(f"  已保存到: train_walking.txt\n")
    
    # 处理 test.txt
    print("处理 test.txt...")
    test_total, test_matched = filter_action(
        'images\\test.txt', 
        'test_walking.txt', 
        action_to_filter
    )
    print(f"  总行数: {test_total}")
    print(f"  匹配行数: {test_matched}")
    print(f"  已保存到: test_walking.txt\n")
    
    # 总结
    print("="*50)
    print("提取完成!")
    print(f"训练集: {train_matched} 条序列")
    print(f"测试集: {test_matched} 条序列")
    print(f"总计: {train_matched + test_matched} 条序列")
    print("="*50)