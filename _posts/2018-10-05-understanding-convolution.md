---
title: 卷积神经网络 - 卷积
teaser: 本文简单介绍卷积基础概念，不同的卷积实现方式和原理，一些卷积使用技巧
category: Computer Vision
tags: [Theory,CNN]
---

# Part 1 简介

### 1.1 我先后在以下场景下见到过卷积

1. 微分方程
2. 傅立叶变换及其应用
3. 概率论
4. 卷积神经网

* 这四种场景表现了卷积在不同领域的应用

### 1.2 相关阅读 

* 推荐一下Christopher Olah 的理解卷积的blog

	<http://colah.github.io/posts/2014-07-Understanding-Convolutions/>

* 数学的论证参考

	<https://www.dsprelated.com/freebooks/mdft/Convolution.html>

### 1.3 本文将介绍卷积在计算机图像处理上的应用，所以限定讨论条件

* 离散的 
* 2-维卷积 (注: 2-维的卷积相当于depth=1 的 3-维的卷积)
* 篇幅有限具体的数学和理解请参考相关阅读

--- 
# Part 2 原理和代码实现

### 2.1 四种方式简单介绍

1. 滑动窗口
* 最直观的方法
* 大规模加速比较困难
* 某些特定场景下速度比较快

2. im2col
* 私以为最主流的实现方式（caffe/MXNet 等使用的是这种方式）
* 一般来说速度较快(空间换时间)
* 卷积过程转化成了 GEMM 过程（被各种极致优化）
* 可以参考

3. FFT
* 傅里叶变换和快速傅里叶变化是在经典图像处理里面经常使用的计算方法
* 在神经网络不常见到
* 卷积模板通常都比较小,例如3×3卷机,这种情况下,FFT的时间开销反而更大
* 会降低精度

4. Winograd
* cudnn 中计算卷积就使用了该方法
* 思想: 更多的加法计算来减少乘法计算
* 原因: 乘法计算的时钟周期数要大于加法计算的时钟周期数）

---
### 2.2 有用的链接

* 卷积的benchmarks 

	<https://github.com/soumith/convnet-benchmarks>

* 介绍GEMM过程

	<https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/>

* im2col的论文

	<https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf>

---	

### 2.3 代码实现（python 2.7 with numpy）

### Talk is cheap show me the code !

### 2.3.1 方法一: 滑动窗口实现
	
```python
import numpy as np 

# calculate output shape 
def calc_shape(input_, kernal_, stride = 1, padding = 0):
	H_out = 1+(input_.shape[0]+2*padding-kernal_.shape[0])/stride
	W_out = 1+(input_.shape[1]+2*padding-kernal_.shape[1])/stride
	return H_out,W_out

def conv2d_naive(input_, kernal_, stride =1, padding = 0):
	# calculate Convolution param 
	out = None
	H,W = input_.shape
	x_pad = np.zeros((H+2*padding,W+2*padding))
	x_pad[padding:padding+H,padding:padding+W] = input_
	H_out,W_out = calc_shape(input_, kernal_, stride, padding)
	out = np.zeros((H_out,W_out))
	# Convolution
	for m in xrange(H_out):
		for n in xrange(W_out):
			out[m,n] = np.sum(x_pad[m*stride:m*stride+kernal_.shape[0],n*stride:n*stride+kernal_.shape[1]] *  kernal_)
	return out

```

### 2.3.2 方法二: im2col方法实现


### 2.3.3 方法三: Winograd方法实现


### 2.3.4 方法四: FTT方法实现
---

# 特征抽取

---

# 引用

---

# 声明
* 转载请联系作者
* 因作者知识水平有限，所述有不确之处欢迎所有人指正批评，共同进步

---
