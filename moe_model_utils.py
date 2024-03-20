
class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        # 初始化共享层，例如Transformer编码器
        self.shared_encoder = TransformerEncoder()

class ExpertModel(nn.Module):
    def __init__(self, task_type):
        super(ExpertModel, self).__init__()
        self.task_type = task_type
        if task_type == 'classification':
            # 分类任务的专家网络
            self.expert = ClassificationHead()
        elif task_type == 'explanation':
            # 解释生成任务的专家网络
            self.expert = ExplanationGenerationHead()

    def forward(self, x):
        return self.expert(x)
        

class MoEModel(nn.Module):
    def __init__(self, experts):
        super(MoEModel, self).__init__()
        self.shared_encoder = SharedEncoder()
        self.experts = nn.ModuleDict(experts)  # 专家网络列表
        self.gate = GatingMechanism()  # 门控机制，简化处理

    def forward(self, inputs, task_type):
        # 通过共享层处理输入
        shared_features = self.shared_encoder(inputs)
        
        # 使用门控机制选择专家
        selected_expert = self.gate(task_type)
        
        # 使用选中的专家处理共享特征
        output = self.experts[selected_expert](shared_features)
        return output

    def adjust_alpha(self, performance_metric_pred, performance_metric_expl):
        # 假设这里使用简单的逻辑来调整alpha，实际应用中可能需要更复杂的策略
        if performance_metric_pred > performance_metric_expl:
            self.alpha -= 0.01  # 减少alpha，给予解释任务更多的重视
        else:
            self.alpha += 0.01  # 增加alpha，给予预测任务更多的重视

        # 确保alpha值在合理的范围内
        self.alpha = max(0, min(1, self.alpha))




