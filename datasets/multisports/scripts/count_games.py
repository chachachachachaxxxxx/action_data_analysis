# 我要读取一个文件夹里的所有MP4文件
# 然后每个文件名去除最后的后缀以及最后的三位占位符
# 然后统计每个经过上述处理的文件名的set集合的数量

import os

def count_games(folder_path):

    game_set = {
        os.path.splitext(file)[0][:-3]
        for file in os.listdir(folder_path)
        if file.endswith('.mp4')
    }
    print(game_set)
    return len(game_set)

# print(os.path.splitext('/storage/wangxinxing/code/action_data_analysis/data/MultiSports/data/trainval/basketball/000001.mp4'))
print(count_games('/storage/wangxinxing/code/action_data_analysis/data/MultiSports/data/trainval/basketball'))