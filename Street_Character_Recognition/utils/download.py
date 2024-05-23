# 下载文件的工具
import urllib.request   

def download_file(url, filename):
    urllib.request.urlretrieve(url, filename)

# 示例用法
download_file('http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json', 'train.json')
