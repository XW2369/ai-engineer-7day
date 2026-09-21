# 00-工程模板 · 从零手写线性回归

7 天 AI 工程师转型冲刺营 **Day 1**。

目标：只用 numpy 从零实现线性回归，并用**工程化手段**证明它是对的——
类型注解、统一日志、单元测试、与 sklearn 对拍，而不是"跑通就行"。

---

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.11.14（`uv` 管理的**项目级** `.venv`） |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| torch | 2.14.0+cpu |
| 工具链 | uv 0.9.9 / ruff / pytest |

## 快速开始

```
uv venv --python 3.11
uv pip install numpy pandas scikit-learn matplotlib seaborn pytest ruff
uv run python -m linreg.linear_regression   # 跑 demo
uv run pytest -v                            # 跑测试 → 6 passed
uv run ruff check .                         # 静态检查 → All checks passed
```

> `pythonpath` 已在 `pyproject.toml` 里配好（`src`），**不需要手动设 `PYTHONPATH`**。

## 目录结构

```
src/linreg/
├── logging_utils.py      统一日志入口，工程代码零 print
├── linear_regression.py  闭式解 + 梯度下降 + 标准化
└── metrics.py            MSE / MAE / R²
tests/
└── test_linear_regression.py   6 个用例，其中 3 条与 sklearn 对拍
```

## 实测结果（2026-09-20）

| 指标 | 数值 |
|---|---|
| 闭式解 vs `sklearn.LinearRegression` 系数最大误差 | **2.44e-15** |
| 梯度下降收敛轮数 | **11 轮**（lr=0.5，tol=1e-8） |
| GD 与闭式解 MSE 差值 | **3.90e-18** |
| 单轮训练耗时 | **0.020 ms**（n=300，d=5，CPU 单线程） |
| pytest | **6 / 6 passed** |
| ruff | **0 error** |

## 三条设计约定

1. **底层函数收到的 `X` 一律是已加偏置列的设计矩阵。**
   截距不单独处理——在 `X` 左侧拼一列 1（`add_bias_term`），它就成了普通系数。
2. **用 `np.linalg.solve`，不用 `np.linalg.inv(X.T @ X)`。**
   $X^TX$ 奇异时，`solve` 直接报错；`inv` 会静默返回一个垃圾解。宁可报错，不要错误答案。
3. **梯度下降前必须先 `standardize`。**
   否则损失函数的等高线是很扁的椭圆，梯度方向不指向谷底，学习率稍大就发散。
   代价是：标准化会让系数跟着变——$\hat\beta$ 不是原坐标系里的值
   （新截距 $=\mu^\top c+b_0$，新斜率 $=\sigma_j c_j$）。

## 学到的东西（踩过的坑）

- shell 里**空格是语法**：`uv --version` ≠ `uv--version`
- **文件名是接口契约**：`tests/` 里 import 的是那个字符串，改文件名等于改 API
- **改完 PATH 必须重开终端**（今天为此浪费了 3 次）
- **`.gitignore` 必须先于 `git add .`**：否则 `.venv` 那 249 个包会被塞进版本库
- **占位符不是值**：`<你的用户名>`、`YOUR_TOKEN` 这类东西要替换，不能照抄（今天踩了 4 次）
