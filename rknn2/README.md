#  环境配置
## 最好保持rknn版本与运行时库以及驱动版本一致
1 需要保证 API    DRV: rknn_server   DRV: rknnrt  三者版本保持一致（三者含义为）

1.1 API版本：API版本是指RKNN提供的编程接口的版本                     （这和在rknn-toolkit2中可找到对应的版本）

1.2 DRV: rknn_server版本：DRV: rknn_server是RKNN的驱动程序        

1.3 DRV: rknnrt版本：DRV: rknnrt是RKNN的运行时库

1.4 可以在这个仓库中找到确定的版本  https://github.com/rockchip-linux/rknpu2 升级2 3 驱动可参考博客   RK3588模型推理总结 - 知乎 (zhihu.com)（step3: 更新板子的rknn_server和librknnrt.so）

# 以1.3版本为例配置rknn-toolkit2环境