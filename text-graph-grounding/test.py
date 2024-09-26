import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--list1', nargs='+', type=float, help='First list of floats')
parser.add_argument('--list2', nargs='+', type=float, help='Second list of floats')

args = parser.parse_args()

print('List 1:', args.list1)  # 输出第一个浮点数列表
print('List 2:', args.list2)  # 输出第二个浮点数列表
