1. 关于训练流程
    1. train backbone的时候使用的temporal weighting可以再显式强调一下，现在那个由浅到深看不太出来是啥
    2. train backbone事实上是train backbone+decoder的级联结构，因为backbone产生的是embedding，但是需要用标签训练，只不过是把decoder fix了，这个地方可以借鉴一些LLM论文里那个freeze冰/training火的标志
    3. backbone其实是用link prediction这个任务预训练过的(warmup)，warmup结束后产生第一次的embedding给decoder作为输入，供decoder的预训练，这么看的话其实也等价于backbone用link prediction预训练过之后从E步开始训练
    4. 显式写出M-step和E-step
2. 美工的地方可以再多加工一下，比如字体，线条的格式和粗细，的现在这个感觉大部分美工有点朴素了
3. 是不是一些地方可以通过颜色来区分，不写文字？比如伪标签和gt用不同的颜色，然后旁边给个图例？感觉每个伪标签都写个pseudo label有点看着不好看，然后那个temporal weighting可以也加个图例，因为其实按method来说这个权重也是随训练时间变化的，我觉得temporal weighting这个地方可以改成，离gt近的一两个点是和gt一样全红，然后离得越远红色越淡
4. 其实结构有点模仿czj，都是左下右上是模型，左上右下是其他模型生成的信息