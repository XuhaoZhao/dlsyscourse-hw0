#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // 遍历所有样本，每次处理一个 batch
    for (size_t i = 0; i < m; i += batch) {
        // 计算当前 batch 的实际大小（处理最后一个可能不足的 batch）
        size_t current_batch_size = std::min(batch, m - i);

        // 临时数组：存储 logits (batch_size x k)
        float* logits = new float[current_batch_size * k];
        // 临时数组：存储梯度 (batch_size x k)
        float* grads = new float[current_batch_size * k];

        /**********************************************
         * 关键点 1：正确处理行主序矩阵乘法 X_batch @ theta
         * X_batch 形状：[batch_size, n] (行主序)
         * theta   形状：[n, k] (行主序)
         * 结果    形状：[batch_size, k] (行主序)
         **********************************************/
        for (size_t b = 0; b < current_batch_size; ++b) {  // 遍历 batch 中的每个样本
            for (size_t c = 0; c < k; ++c) {              // 遍历每个类别
                logits[b * k + c] = 0.0f;                 // 初始化 logit
                // 矩阵乘法：X 的第 (i+b) 行 与 theta 的第 c 列 的点积
                for (size_t f = 0; f < n; ++f) {          // 遍历每个特征
                    // X 访问：第 (i+b) 行，第 f 列 -> X[(i+b)*n + f]
                    // Theta 访问：第 f 行，第 c 列 -> theta[f*k + c] (行主序)
                    logits[b * k + c] += X[(i + b) * n + f] * theta[f * k + c];
                }
            }
        }

        /**********************************************
         * 关键点 2：softmax 计算（注意数值稳定性）
         **********************************************/
        for (size_t b = 0; b < current_batch_size; ++b) {
            // 1. 找到最大 logit 值（防止指数爆炸）
            float max_logit = logits[b * k];
            for (size_t c = 1; c < k; ++c) {
                if (logits[b * k + c] > max_logit) {
                    max_logit = logits[b * k + c];
                }
            }

            // 2. 计算指数和
            float exp_sum = 0.0f;
            for (size_t c = 0; c < k; ++c) {
                logits[b * k + c] = expf(logits[b * k + c] - max_logit); // 数值稳定
                exp_sum += logits[b * k + c];
            }

            // 3. 计算 softmax 概率和梯度
            for (size_t c = 0; c < k; ++c) {
                logits[b * k + c] /= exp_sum;          // 概率值
                grads[b * k + c] = logits[b * k + c];   // 梯度初始化为概率
            }

            // 4. 正确类别的梯度调整
            unsigned char true_class = y[i + b];        // 当前样本的真实标签
            grads[b * k + true_class] -= 1.0f;          // 梯度 = 概率 - one-hot
        }

        /**********************************************
         * 关键点 3：梯度计算 (X_batch.T @ grads) 
         * X_batch.T 形状：[n, batch_size] (列主序的隐式转置)
         * grads     形状：[batch_size, k] (行主序)
         * 结果      形状：[n, k] (行主序)
         **********************************************/
        for (size_t f = 0; f < n; ++f) {            // 遍历每个特征（对应转置后的行）
            for (size_t c = 0; c < k; ++c) {        // 遍历每个类别
                float grad_sum = 0.0f;
                // 计算 X 第 f 列与 grads 第 c 列的点积
                for (size_t b = 0; b < current_batch_size; ++b) {
                    // X_batch 转置后的访问：原列 f -> 现行 f，元素为 X[(i+b)*n + f]
                    grad_sum += X[(i + b) * n + f] * grads[b * k + c];
                }
                // 更新 theta：注意学习率和 batch 大小的缩放
                theta[f * k + c] -= lr * grad_sum / current_batch_size;
            }
        }

        // 释放临时内存
        delete[] logits;
        delete[] grads;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
