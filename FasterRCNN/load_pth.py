import os
import datetime

import torch
# from torchsummary import summary
# from torchstat import stat

state_dict_0 = torch.load("./save_weights/resNetFpn-model-0.pth")
state_dict_1 = torch.load("./save_weights/resNetFpn-model-1.pth")
state_dict_2 = torch.load("./save_weights/resNetFpn-model-2.pth")
state_dict_3 = torch.load("./save_weights/resNetFpn-model-3.pth")
state_dict_4 = torch.load("./save_weights/resNetFpn-model-4.pth")

state_dict_5 = torch.load("./save_weights/resNetFpn-model-5.pth")
# # print(state_dict_5)
# # print(type(state_dict_5)) # <class 'dict'>
# # print(len(state_dict_5))
# #
# for k in state_dict_5.keys():
#     print(k)
#
# # print(state_dict_5["model"])
#
# # for i in state_dict_5:
# #     print(i)
# #     print(type(state_dict_5[i]))
# #     print(len(state_dict_5[i]))
# #     print("aa:", state_dict_5[i].data.size())
# #     print("bb:", state_dict_5[i].requires_grad)
# #     break
#
# # for key, value in state_dict_5["model"].items():
# #     print(key, value.size(), sep="&")
# #
# # print(len(state_dict_5["model"]))
#
# # print(len(state_dict_5["optimizer"])) # 2
# # print(state_dict_5["optimizer"])
# # for key, value in state_dict_5["optimizer"].items():
# #     print(key) # , len(value), sep=" "
# # state 156
# # param_groups 1
#
# # print(len(state_dict_5["lr_scheduler"]))
# # print(type(state_dict_5["lr_scheduler"]))
#
# for key, value in state_dict_5["optimizer"]["state"].items():
#     print(key, value['momentum_buffer'].size(), sep="&")
#
# print(state_dict_5["optimizer"]["param_groups"])

for key, value in state_dict_0["lr_scheduler"].items():
    print(key, value, sep="&")  #
print(state_dict_0["lr_scheduler"])
print(state_dict_0["epoch"])


for key, value in state_dict_1["lr_scheduler"].items():
    print(key, value, sep="&")  #
print(state_dict_1["lr_scheduler"])
print(state_dict_1["epoch"])


for key, value in state_dict_2["lr_scheduler"].items():
    print(key, value, sep="&")  #
print(state_dict_2["lr_scheduler"])
print(state_dict_2["epoch"])


for key, value in state_dict_3["lr_scheduler"].items():
    print(key, value, sep="&")  #
print(state_dict_3["lr_scheduler"])
print(state_dict_3["epoch"])

for key, value in state_dict_4["lr_scheduler"].items():
    print(key, value, sep="&")  #
print(state_dict_4["lr_scheduler"])
print(state_dict_4["epoch"])

for key, value in state_dict_5["lr_scheduler"].items():
    print(key, value, sep="&")  #
print(state_dict_5["lr_scheduler"])
print(state_dict_5["epoch"])



# print(torch.__version__)
#
# print(torch.cuda.is_available())
#
# 但是由于C盘空间小，若不想把虚拟环境放在默认的c盘下该怎么办呢？
#
# 通过查阅anaconda的文档，发现是可以进行指定路径安装的。可以输入如下命令进行查看：
#
# conda create --help
#
# 安装虚拟环境到指定路径的命令如下：
#
# conda create --prefix=D:\python38-env\pytorch_gpu python=3.8
#
# 上面的命令中， 路径D:\python38-env\是先建好的文件夹(你也可以不建立，没有会自动建立），py35是需要安装的虚拟环境名称。
# 请注意，安装完成后，虚拟环境的全称包含整个路径，为D:\python38-env\pytorch_gpu。激活指定路径下的虚拟环境的命令如下：
#
# activate D:\python38-env\pytorch_gpu
#
# 想要删除指定路径下的虚拟环境，使用如下的命令：
#
# conda remove --prefix=D:\python38-env\pytorch_gpu --all


