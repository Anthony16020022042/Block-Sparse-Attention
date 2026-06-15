# BlockSparseAttention for Ascend NPU

本仓库提供 block sparse-attn 在华为昇腾 NPU 上的实现。

## 安装

**环境要求：**

- CANN
- PyTorch 2.1 及以上
- torch_npu
- `packaging` Python 包（`pip install packaging`）
- `psutil` Python 包（`pip install psutil`）
- `ninja` Python 包（`pip install ninja`）
- Linux

**安装步骤：**

设置环境变量：
```

```

从源码编译：
```bash
git clone https://github.com/your-repo/Block-Sparse-Attention.git
cd Block-Sparse-Attention
git submodule update --init --recursive
python setup.py install
```

## 测试

运行测试：
```bash
pytest -q -s block_sparse_tests/test_bsa_attn_npu.py
```
