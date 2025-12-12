import os

# 配置：指定要处理的标签文件夹路径
label_folder = r"C:\Users\wyk31\Desktop\nn论文\0-1_change"  # ← 修改为你的 .txt 文件所在目录


def swap_first_char(line):
    """将一行中第一个字符（类别ID）0变1，1变0，其他不变"""
    line = line.strip()
    if not line:
        return line  # 空行直接返回

    first_char = line[0]
    rest = line[1:]

    if first_char == '0':
        return '1' + rest
    elif first_char == '1':
        return '0' + rest
    else:
        # 如果不是0或1（比如有其他类别），可选择保留原样或报错
        print(f"⚠️ 警告：发现非0/1类别ID '{first_char}'，已跳过该行。")
        return line  # 保持不变


# 遍历文件夹中所有 .txt 文件
for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(label_folder, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            new_line = swap_first_char(line)
            new_lines.append(new_line + '\n')  # 保留换行

        # 覆盖原文件（建议先备份！）
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

print(f"✅ 已完成对文件夹 '{label_folder}' 中所有 .txt 文件的类别标签交换（0↔1）。")