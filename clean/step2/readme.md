这里的脚本是为了过滤非战术视角的情况

先提取管，然后创建一个图片文件夹，交由模型过滤，得到的结果再进行解析然后调用tube级别的过滤命令过滤即可
```shell
python multisports_clean_extractor.py
zsh batch_clean.sh
python extract_zero_pairs.py
zsh remove_tubes.sh
```