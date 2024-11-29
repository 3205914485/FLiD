# Some idea & notes of NcEM:

## -11.25-
### 1. 故事

故事出发点：
    - **有时间戳的 静态图结构数据集** 缺乏对时间信息的利用（我们采用动态建模手段）
    - **有时间标签的 动态图结构数据集** 动态标签的采集困难与质量问题（我们采用半监督建模手段）

Based on the
- timestamp
- dynamic-transfer
- dynamic-labelsv

We list:
- 1.  Static-Graph                      (000)
- 2.  Static-Graph with only ts         (100)
- 3. **Static-Graph with ts & dy_trans   (110)**
- 4.  Dynmaic-Graph wo/ dy_trans        (101)
- 5. **Dynamic-Graph                     (111)**    

关于节点分类：
    我们利用最后时刻的节点标签进行分类。可以通过递归设计最后时刻，来对整个数据集进行分类，并且训练好后进行推理时可扩展性优秀
    我们设计了新的划分方式

### 2. 实验
    3型只对比ED；
    5型对比ED、Sup；
        还可以利用生成的伪标签进行有监督训练来证明效果
    
    pb 实验进行调参
    