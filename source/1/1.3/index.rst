1.3 深度学习框架安装配置(Pytorch)
=======================================

首先从PyTorch官网的下载网址[10]下载合适的PyTorch版本到D盘（Linux则对应到其他相应位置），本书下载的是cu116/torch-1.12.0%2Bcu116-cp38- cp38-win_amd64.whl（即CUDA版本为11.6，PyTorch版本为1.12，Python版本为3.8）。

打开MiniConda，进入命令行模式，通过路径切换命令，将当前路径切换到PyTorch安装文件所在位置（此处假设安装文件下载到D:\ToLMC\文件夹）, 然后输入如图1-8所示的安装命令以安装PyTorch包。

.. image:: /_static/1/1.3/1-8.png
    :alt: 图 1-8 虚拟环境中切换到软件所在盘并安装

待安装结束后，若出现“successfully installed …”等提示信息，说明安装成功；否则说明安装失败，需重新安装。

安装成功后，可通过torch.__version__命令验证是否安装成功，若打印出相应版本号则表示安装成功，如图1-9所示；若提示“ModuleNotFoundError…”则表示安装失败，如图1-10所示。

.. image:: /_static/1/1.3/1-9.png
    :alt: 图 1-8 安装成功后提示

.. image:: /_static/1/1.3/1-10.png
    :alt: 图 1-8 安装失败后提示
