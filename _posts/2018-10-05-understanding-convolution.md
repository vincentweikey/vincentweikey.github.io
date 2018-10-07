---
title: 卷积神经网络基本概念（卷积篇 01）
teaser: 本文简单介绍卷积基础概念，不同的卷积实现方式和原理
category: Computer Vision
tags: [Theory,CNN]
---

# Part 1 简介

### 1.1 我先后在以下场景下见到过卷积

1. 微分方程
2. 傅立叶变换及其应用
3. 概率论
4. 卷积神经网

### 1.2 相关阅读 

* 推荐一下 Christopher Olah 的理解卷积的blog

	<http://colah.github.io/posts/2014-07-Understanding-Convolutions/>

* 数学论证

	<https://www.dsprelated.com/freebooks/mdft/Convolution.html>

### 1.3 本文将介绍卷积在计算机图像处理上的应用，所以限定讨论条件

* 离散的 
* 2-维卷积(注:2-维的卷积相当于depth=1的3-维的卷积)
* 篇幅有限具体的数学和理解请参考相关阅读

--- 
# Part 2 原理和代码实现

### 2.1 四种方式简单介绍

### 无损精度算法

#### 2.1.1滑动窗口

* 最直观的方法
* 大规模加速比较困难
* 某些特定场景下速度比较快

#### 2.1.2 im2col

* 私以为最主流的实现方式（caffe/MXNet 等使用的是这种方式）
* 一般来说速度较快(空间换时间)
* 卷积过程转化成了GEMM过程（被各种极致优化)
* 算法：
	1. 将没有个需要做卷积的矩阵平摊为一个一维向量
	2. 将所有的向量组成新的矩阵
	3. 新矩阵和kernal做矩阵乘法
	4. 得到结果

### 有损精度算法

* FFT/Winograd的卷积算法，它们都是通过:

	1. 某种线性变换将feature map和卷积核变换到另外一个域.
	2. 空间域下的卷积在这个域下变为逐点相乘.
	3. 通过另一个线性变换将结果变换到空间域.

* 有幸听过王晋玮的关于深度学习加速的 paper reading,其中有提及到:  FFT需要复数乘法，如果没有特殊指令支持的话需要用实数乘法来模拟,实数的浮点计算量可能下降的不多,因此FFT也没有Winograd实用.

#### 2.1.3 FFT

* 傅里叶变换和快速傅里叶变化是在经典图像处理里面经常使用的计算方法.
* 卷积模板通常都比较小,例如3×3卷机,这种情况下,FFT的时间开销反而更大.
* 具体而言FFT将空间意义上的实数变换到频域上的复数,最后在复数上做逐点相乘,然后再把这个频率的复数变化为这个空间域的实数.
* FFT卷积采用傅里叶变换处理 feature map和卷积核,傅里叶逆变换处理结果.

#### 2.1.4 Winograd
* cudnn中计算卷积就使用了该方法.
* CNN里面越来越多的1×1卷积和depthwise卷积被加入,Winograd卷积的价值也越来越小了.
* 可以实现极高的一个加速比,举个例子,Winograd变换对于3×3卷积,最高可以实现9倍的加速比,但精度损失严重.


---
### 2.2 有用的链接

* 卷积的benchmarks 

	<https://github.com/soumith/convnet-benchmarks>

* 介绍GEMM过程

	<https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/>

* im2col的论文

	<https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf>

* winograd的python实现
	
	<https://github.com/andravin/wincnn>

---	

### 2.3 代码实现（python 2.7 with numpy）

### ------ 
### Talk is cheap show me the code !
### ------

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


```python 
import numpy as np 

# calculate output shape 
def calc_shape(input_, kernal_, stride = 1, padding = 0):
	H_out = 1+(input_.shape[0]+2*padding-kernal_.shape[0])/stride
	W_out = 1+(input_.shape[1]+2*padding-kernal_.shape[1])/stride
	return H_out,W_out

def im2col(input_, kernal_):
	# calculate param of col matrix
	X_ = input_.shape[0]-kernal_.shape[0]+1
	Y_ = input_.shape[1]-kernal_.shape[1]+1
	output_col = np.empty((kernal_.shape[0]*kernal_.shape[1], X_*Y_))
	# im2col
	for i in range(Y_):
		for j in range(X_):
			output_col[:,i*Y_+j] = input_[j:j+kernal_.shape[0],i:i+kernal_.shape[1]].ravel(order='F')
	return output_col

def col2im(input_, kernal_ ,col_matrix):
	output_ = np.zeros(input_.shape) 
	weight_ = np.zeros(input_.shape)
	col = 0
	X_ = input_.shape[0] - kernal_.shape[0] + 1
	Y_ = input_.shape[1] - kernal_.shape[1] + 1
	for i in range(Y_):
		for j in range(X_):
			output_[j:j+kernal_.shape[0],i:i+kernal_.shape[1]]+= col_matrix[:,col].reshape(kernal_.shape, order='F')
			weight_[j:j+kernal_.shape[0],i:i+kernal_.shape[1]] += np.ones(kernal_.shape)
			col+=1
	return output_/weight_

def conv2d_im2col(input_, kernal_, stride =1, padding = 0):
	# stride must = 1 in this code 
	# this code is only show how im2col work
	# if you want stride = other number , change the np.dot
	H,W = input_.shape
	x_pad = np.zeros((H+2*padding ,W+2*padding))
	x_pad[padding:padding+H,padding:padding+W] = input_

	# im2col processing and dot col and kernal
	output_col = im2col(x_pad, kernal_)
	col_matrix = np.dot(kernal_.reshape(kernal_.shape[0]*kernal_.shape[1],order='F'),output_col)
	
	# reshape to the final status
	H_out,W_out = calc_shape(input_, kernal_, stride, padding)
	out = np.zeros((H_out,W_out))
	return col_matrix.reshape(out.shape[0],out.shape[1],order='F')

```
### 2.3.3 方法三: FFT方法实现

* 使用到numpy的FFT库

	<https://docs.scipy.org/doc/numpy/reference/routines.fft.html>

```python 

def FFT_convolve(input_,kernal_):
	# calculate the FFT trans size
	T_size = np.array(input_.shape)+np.array(kernal_.shape)-1
	# ceil doc : https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ceil.html
	F_size = 2**np.ceil(np.log2(T_size)).astype(int)
	F_slice = tuple([slice(0, int(item)) for item in F_size])

	new_input_ = np.fft.fft2(input_, F_size)
	new_kernal_ = np.fft.fft2(kernal_, F_size)
	output_ = np.fft.ifft2(new_input_*new_kernal_)[F_slice].copy()
	# output is a expand matrix which bigger than the result you suppose
	return np.array(output_.real, np.int32)

```

### 2.3.4 Final : 运行方法

```python 

def demo_1():
	input_ = np.diag([1, 10, 3, 5, 1])
	kernal_ = np.diag([1, 2, 1])
	return conv2d_naive(input_, kernal_, stride =1, padding = 0)

def demo_2():
	input_ = np.diag([1, 1, 1, 1, 1])
	kernal_ = np.diag([1, 1, 1])
	# stride must equal 1 in my demo code
	return conv2d_im2col(input_, kernal_, stride =1, padding = 0)

def demo_3():
	input_ = np.diag([1, 10, 3, 5, 1])
	kernal_ = np.diag([1, 2, 1,])
	return FFT_convolve(input_, kernal_)

print demo_1()
print demo_2()
print demo_3()

``` 

---
# Part 3 声明
* 转载请联系作者，虽然不联系也没什么问题～
* 邮箱右上角contact
* 因作者知识水平有限，所述有不确之处欢迎指正批评，感激笔芯～

---
