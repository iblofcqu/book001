# 开发人员本地调试

以下内容基于已经`clone`准备好的本地仓库，搭建调试环境

## 选择 python 环境

选择 `python>3.9` 的环境

## 安装调试所需要的第三方库

```shell
pip install -r requirements-dev.txt
```

**注**：

`requirements.txt`文件用于`readthedoc`制作静态文件.

`requirements-dev.txt` 中包括了各个示例代码中需要的工具库.

## 生成静态文件

```shell
sphinx-build -b html source build
```

通过以上命令,可以在`build` 目录下生成`html`静态文件,通过浏览器打开`index.html`
即可预览。

## 实时预览

```shell
sphinx-autobuild source build
```

该命令会依次完成：制作静态文件，运行基于http的静态文件服务器，以及监听source
下的markdown 文件。

该命令比较吃性能,请酌情选择.

## 数据集准备

各章节数据集均放在 章节目录`data`下,需要参照各个章节内容下载准备数据集

## 其他注意事项

1. Docstring 风格选择 `Google`
