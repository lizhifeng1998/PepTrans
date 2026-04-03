import os
import sys
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

# --------------------------
# 1. 配置参数
# --------------------------
# 氨基酸表（20种标准氨基酸 + 未知字符）
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY-'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}

# 模型参数
MAX_SEQ_LEN = 20  # 肽序列最大长度
EMBEDDING_DIM = 64  # 嵌入维度
NUM_HEADS = 8  # 注意力头数
NUM_ENCODER_LAYERS = 2  # Transformer编码器层数
FFN_HIDDEN_DIM = 128  # 前馈网络隐藏层维度
DROPOUT = 0.1  # dropout率

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# 2. 数据预处理
# --------------------------
class PeptideDataset(Dataset):
    """肽序列数据集类"""
    def __init__(self, sequences, properties, max_len=MAX_SEQ_LEN):
        self.sequences = sequences
        self.properties = np.array(properties).reshape(-1, 1)
        self.max_len = max_len
        
        # 标准化性质值（回归任务关键）
        self.scaler = StandardScaler()
        self.properties_scaled = self.scaler.fit_transform(self.properties)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        prop = self.properties_scaled[idx]
        
        # 1. 序列编码：将氨基酸转换为索引
        encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['-']) for aa in seq]
        
        # 2. 序列填充/截断到固定长度
        if len(encoded) < self.max_len:
            encoded += [AA_TO_IDX['-']] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        
        # 3. 创建注意力掩码（忽略填充位）
        attention_mask = [1] * len(seq) + [0] * (self.max_len - len(seq)) if len(seq) < self.max_len else [1]*self.max_len
        
        return {
            'sequence': torch.tensor(encoded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'property': torch.tensor(prop, dtype=torch.float32)
        }
# ===================== 1. 定义氨基酸物化性质（已归一化） ===================== 
# 包含20种常见氨基酸的4个核心物化特征：疏水性、电荷、侧链体积、极性
# 特征值已做min-max归一化到[0,1]区间
AMINO_ACID_PROPS = {
    'A': [0.62, 0.0, 0.03, 0.0],    # 丙氨酸: 疏水性、电荷、体积、极性
    'C': [0.29, 0.0, 0.12, 0.1],    # 半胱氨酸
    'D': [-0.90, -1.0, 0.18, 1.0],  # 天冬氨酸
    'E': [-0.74, -1.0, 0.24, 1.0],  # 谷氨酸
    'F': [1.19, 0.0, 0.41, 0.0],    # 苯丙氨酸
    'G': [0.48, 0.0, 0.0, 0.0],     # 甘氨酸
    'H': [-0.40, 0.5, 0.27, 0.8],   # 组氨酸（中性pH下部分带正电
    'I': [1.38, 0.0, 0.31, 0.0],    # 异亮氨酸
    'K': [-1.50, 1.0, 0.32, 1.0],   # 赖氨酸
    'L': [1.06, 0.0, 0.31, 0.0],    # 亮氨酸
    'M': [0.64, 0.0, 0.25, 0.0],    # 甲硫氨酸
    'N': [-0.78, 0.0, 0.20, 1.0],   # 天冬酰胺
    'P': [0.12, 0.0, 0.13, 0.0],    # 脯氨酸
    'Q': [-0.85, 0.0, 0.26, 1.0],   # 谷氨酰胺
    'R': [-2.53, 1.0, 0.36, 1.0],   # 精氨酸
    'S': [-0.18, 0.0, 0.06, 0.9],   # 丝氨酸
    'T': [-0.05, 0.0, 0.14, 0.9],   # 苏氨酸
    'V': [1.08, 0.0, 0.22, 0.0],    # 缬氨酸
    'W': [0.81, 0.0, 0.47, 0.2],    # 色氨酸
    'Y': [0.26, 0.0, 0.38, 0.8],    # 酪氨酸
    '-': [0.00, 0.0, 0.0, 0.0],     #
}

# --------------------------
# 3. Transformer模型定义
# --------------------------
class PeptideTransformerRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 氨基酸嵌入层
        self.embedding = nn.Embedding(len(AMINO_ACIDS), EMBEDDING_DIM)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FFN_HIDDEN_DIM,
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True  # 输入格式：(batch, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        
        # 回归头：将Transformer输出映射到单个数值
        self.regression_head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, FFN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_HIDDEN_DIM, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # 1. 嵌入层
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # 2. Transformer编码
        # 将attention_mask转换为Transformer需要的格式（True表示忽略）
        mask = (attention_mask == 0)
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=mask)  # (batch, seq_len, embed_dim)
        
        # 3. 聚合序列特征（使用CLS方式：取第一个位置的输出）
        cls_output = encoded[:, 0, :]  # (batch, embed_dim)
        
        # 4. 回归预测
        output = self.regression_head(cls_output)  # (batch, 1)
        
        return output

class PeptideTransformerWithProps(nn.Module):
    def __init__(self,
                 max_length=20,
                 vocab_size=len(AMINO_ACIDS),
                 embedding_dim=128,
                 num_heads=8,
                 ffn_hidden_dim=256,
                 dropout=0.1,
                 num_encoder_layers=3,
                 num_props=4):
        super().__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_props = num_props
        # 1. 氨基酸离散嵌入层
        self.aa_embedding = nn.Embedding(vocab_size, embedding_dim)
        # 2. 物化特征处理层：将数值特征映射到embedding_dim维度
        self.prop_linear = nn.Sequential(
            nn.Linear(num_props, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 3. 特征融合层：拼接后投影（可选：也可直接相加，需保证维度一致
        self.fusion_linear = nn.Linear(embedding_dim * 2, embedding_dim)
        # 4. 位置编码（预留CLS位置
        self.position_encoding = nn.Parameter(
            torch.zeros(1, max_length + 1, embedding_dim)
        )
        nn.init.normal_(self.position_encoding, mean=0, std=0.02)
        # 5. CLS标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        # 6. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        # 7. 回归头
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, 1)
        )
        # 预先生成物化特征矩阵（避免每次forward都查表，提升效率
        self.prop_matrix = self._build_prop_matrix()

    def _build_prop_matrix(self):
        """构建氨基酸索引到物化特征的矩阵，shape: (vocab_size, num_props)"""
        prop_list = []
        for aa in AMINO_ACIDS:
            prop_list.append(AMINO_ACID_PROPS[aa])
        prop_matrix = torch.tensor(prop_list, dtype=torch.float32)
        # 转为nn.Parameter，确保设备同步（但不参与梯度更新）
        return nn.Parameter(prop_matrix, requires_grad=False)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        # ========== 步骤1：获取氨基酸嵌入 ==========
        # input_ids: (batch_size, seq_len)
        aa_embedded = self.aa_embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        # ========== 步骤2：获取并处理物化特征 ==========
        # 从预构建的矩阵中取物化特征: (batch_size, seq_len, num_props)
        prop_features = self.prop_matrix[input_ids]
        # 映射到embedding_dim维度: (batch_size, seq_len, embedding_dim)
        prop_embedded = self.prop_linear(prop_features)
        # ========== 步骤3：特征融合 ==========
        # 拼接离散嵌入和物化特征: (batch_size, seq_len, 2*embedding_dim)
        fused_embedded = torch.cat([aa_embedded, prop_embedded], dim=-1)
        # 投影回embedding_dim维度: (batch_size, seq_len, embedding_dim)
        fused_embedded = self.fusion_linear(fused_embedded)
        # ========== 步骤4：添加CLS和位置编码（与原逻辑一致） ==========
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embedding_dim)
        fused_embedded = torch.cat([cls_tokens, fused_embedded], dim=1)  # (batch_size, seq_len+1, embedding_dim)
        # 添加位置编码（避免越界
        pos_encoding = self.position_encoding[:, :fused_embedded.size(1), :]
        fused_embedded = fused_embedded + pos_encoding
        # ========== 步骤5：调整注意力掩码 ==========
        cls_mask = torch.ones(batch_size, 1, device=fused_embedded.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        src_key_padding_mask = (attention_mask == 0)
        # ========== 步骤6：Transformer编码 + 回归预测 ========== 
        encoded = self.transformer_encoder(fused_embedded, src_key_padding_mask=src_key_padding_mask)
        cls_output = encoded[:, 0, :]  # 取CLS输出
        # output = self.regression_head(cls_output).squeeze(-1)
        output = self.regression_head(cls_output)
        return output

# --------------------------
# 4. 训练和验证函数
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs,
                scaler=None, save_path='best_peptide_model.pth'):
    """训练模型并验证

    Args:
        scaler: StandardScaler实例，用于保存到checkpoint中
        save_path: 最佳模型保存路径
    """
    model.to(DEVICE)
    best_val_rmse = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []

        for batch in train_loader:
            input_ids = batch['sequence'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['property'].to(DEVICE)

            # 前向传播
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['sequence'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                targets = batch['property'].to(DEVICE)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)

                val_losses.append(loss.item())
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # 计算验证指标
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)

        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'best_rmse': best_val_rmse
            }, save_path)

        # 打印日志
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}')
        print('-' * 50)
    print(best_val_rmse)

# --------------------------
# 5. 预测函数
# --------------------------
def predict_peptide_property(model_path, sequences):
    """预测新肽序列的性质"""
    # 加载模型和scaler
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    # model = PeptideTransformerRegressor()
    model = PeptideTransformerWithProps()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    scaler = checkpoint['scaler']
    
    # 预处理序列
    encoded_sequences = []
    attention_masks = []
    
    for seq in sequences:
        # 编码
        encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['-']) for aa in seq]
        # 填充/截断
        if len(encoded) < MAX_SEQ_LEN:
            encoded += [AA_TO_IDX['-']] * (MAX_SEQ_LEN - len(encoded))
        else:
            encoded = encoded[:MAX_SEQ_LEN]
        # 注意力掩码
        attention_mask = [1] * len(seq) + [0] * (MAX_SEQ_LEN - len(seq)) if len(seq) < MAX_SEQ_LEN else [1]*MAX_SEQ_LEN
        
        encoded_sequences.append(encoded)
        attention_masks.append(attention_mask)
    
    # 转换为tensor
    input_ids = torch.tensor(encoded_sequences, dtype=torch.long).to(DEVICE)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long).to(DEVICE)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_masks)
    
    # 反标准化得到原始尺度的预测值
    predictions = scaler.inverse_transform(outputs.cpu().numpy())
    
    return predictions.flatten()

# --------------------------
# 6. 命令行入口
# --------------------------
def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='肽序列Transformer模型：训练、验证、预测、外推',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ===== 子命令（四个功能模式） =====
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    subparsers.required = True

    # ---------- 共享参数（通过 parent parser） ----------
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('--data', type=str, default='stabil041.txt',
                        help='输入数据文件路径（每行: 序列 性质值），默认: stabil041.txt')
    parent.add_argument('--model-path', type=str, default='best_peptide_model.pth',
                        help='模型文件路径（训练时为保存路径，其他模式为加载路径），默认: best_peptide_model.pth')
    parent.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'批次大小，默认: {BATCH_SIZE}')
    parent.add_argument('--seed', type=int, default=42,
                        help='随机种子，默认: 42')
    parent.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例，默认: 0.8')

    # ---------- 1. train（训练） ----------
    train_parser = subparsers.add_parser('train', parents=[parent],
                                         help='训练模型')
    train_parser.add_argument('--epochs', type=int, default=EPOCHS,
                              help=f'训练轮数，默认: {EPOCHS}')
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                              help=f'学习率，默认: {LEARNING_RATE}')
    train_parser.add_argument('--weight-decay', type=float, default=1e-5,
                              help='权重衰减，默认: 1e-5')
    train_parser.add_argument('--no-overwrite', action='store_true',
                              help='若模型文件已存在则报错退出（防止覆盖）')

    # ---------- 2. validate（验证） ----------
    val_parser = subparsers.add_parser('validate', parents=[parent],
                                       help='在验证集上评估模型')
    val_parser.add_argument('--threshold', type=float, default=None,
                            help='筛选验证集的性质值阈值（仅验证 > threshold 的样本），默认: 不筛选')

    # ---------- 3. predict（预测） ----------
    pred_parser = subparsers.add_parser('predict', parents=[parent],
                                        help='对指定序列进行预测')
    pred_parser.add_argument('--sequences', type=str, nargs='+', default=None,
                             help='直接指定要预测的肽序列（空格分隔）')
    pred_parser.add_argument('--seq-file', type=str, default=None,
                             help='从文件读取待预测序列（每行一个序列）')
    pred_parser.add_argument('--filter-threshold', type=float, default=-0.8,
                             help='从数据文件中筛选验证集外序列的性质值阈值，默认: -0.8')
    pred_parser.add_argument('--output', type=str, default=None,
                             help='预测结果输出文件路径（默认: 输出到stdout）')

    # ---------- 4. extrapolate（外推） ----------
    ext_parser = subparsers.add_parser('extrapolate', parents=[parent],
                                       help='随机生成序列并筛选高性质值候选')
    ext_parser.add_argument('--threshold', type=float, default=0.9,
                            help='外推筛选阈值（预测值 > threshold 的序列被保留），默认: 0.9')
    ext_parser.add_argument('--seq-len-min', type=int, default=8,
                            help='外推序列最小长度，默认: 8')
    ext_parser.add_argument('--seq-len-max', type=int, default=13,
                            help='外推序列最大长度（不含），默认: 13')
    ext_parser.add_argument('--target-count', type=int, default=100,
                            help='每个长度目标筛选数量，默认: 100')
    ext_parser.add_argument('--max-batch', type=int, default=40000,
                            help='外推时每批最大序列数，默认: 40000')
    ext_parser.add_argument('--output', type=str, default=None,
                            help='外推结果输出文件路径（默认: 输出到stdout）')

    return parser.parse_args()


def load_data(data_path):
    """从文件加载序列和性质值数据"""
    sequences = []
    properties = []
    dic = {}
    with open(data_path, 'r') as f:
        for line in f.readlines():
            s = line.strip().split()
            if len(s) >= 2:
                sequences.append(s[0])
                properties.append(float(s[1]))
                dic[s[0]] = float(s[1])
    return sequences, properties, dic


def setup_seed(seed):
    """设置随机种子以确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(dataset, dic, train_ratio, batch_size):
    """构建训练/验证数据加载器"""
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 提取训练集序列用于加权采样
    train_seq = [
        ''.join([IDX_TO_AA[int(x)] for x in train_dataset[i]['sequence']]
                [:sum(train_dataset[i]['attention_mask'])])
        for i in range(len(train_dataset))
    ]
    train_pro = [dic[train_seq[i]] * len(train_seq[i]) for i in range(len(train_dataset))]
    sampler = WeightedRandomSampler(weights=train_pro, num_samples=len(train_pro), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_seq


def run_train(args):
    """训练模式"""
    if args.no_overwrite and os.path.exists(args.model_path):
        print(f'错误: 模型文件 {args.model_path} 已存在，使用 --no-overwrite 时不允许覆盖。', file=sys.stderr)
        sys.exit(1)

    sequences, properties, dic = load_data(args.data)
    dataset = PeptideDataset(sequences, properties)
    train_loader, val_loader, _ = build_dataloaders(dataset, dic, args.train_ratio, args.batch_size)

    model = PeptideTransformerWithProps()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f'开始训练: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}')
    print(f'数据文件: {args.data}, 模型保存路径: {args.model_path}')
    print(f'训练集: {len(train_loader.dataset)} 条, 验证集: {len(val_loader.dataset)} 条')
    print('=' * 50)

    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs,
                scaler=dataset.scaler, save_path=args.model_path)


def run_validate(args):
    """验证模式：加载已训练模型，在验证集上评估"""
    sequences, properties, dic = load_data(args.data)
    dataset = PeptideDataset(sequences, properties)
    _, val_loader, train_seq = build_dataloaders(dataset, dic, args.train_ratio, args.batch_size)

    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    model = PeptideTransformerWithProps()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    scaler = checkpoint['scaler']

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['sequence'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['property'].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

    # 反标准化
    val_preds_orig = scaler.inverse_transform(np.array(val_preds))
    val_targets_orig = scaler.inverse_transform(np.array(val_targets))

    rmse = np.sqrt(mean_squared_error(val_targets_orig, val_preds_orig))
    r2 = r2_score(val_targets_orig, val_preds_orig)

    print(f'验证集评估结果:')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  R²:   {r2:.4f}')
    print(f'  样本数: {len(val_preds)}')

    if args.threshold is not None:
        # 筛选特定范围的样本单独评估
        mask = val_targets_orig.flatten() > args.threshold
        if mask.sum() > 0:
            sub_rmse = np.sqrt(mean_squared_error(val_targets_orig[mask], val_preds_orig[mask]))
            sub_r2 = r2_score(val_targets_orig[mask], val_preds_orig[mask])
            print(f'\n  筛选 > {args.threshold} 的样本:')
            print(f'    RMSE: {sub_rmse:.4f}')
            print(f'    R²:   {sub_r2:.4f}')
            print(f'    样本数: {mask.sum()}')
        else:
            print(f'\n  没有性质值 > {args.threshold} 的样本。')


def run_predict(args):
    """预测模式"""
    # 确定待预测序列来源
    if args.sequences:
        # 直接从命令行指定
        test_sequences = args.sequences
        dic = {}
    elif args.seq_file:
        # 从文件读取
        test_sequences = []
        with open(args.seq_file, 'r') as f:
            for line in f:
                seq = line.strip().split()[0]
                if seq:
                    test_sequences.append(seq)
        dic = {}
    else:
        # 从数据文件中筛选（原始逻辑：验证集中满足阈值的序列）
        sequences, properties, dic = load_data(args.data)
        dataset = PeptideDataset(sequences, properties)
        _, _, train_seq = build_dataloaders(dataset, dic, args.train_ratio, args.batch_size)
        test_sequences = [k for k in dic.keys() if dic[k] > args.filter_threshold and k not in train_seq]

    if not test_sequences:
        print('没有找到待预测的序列。', file=sys.stderr)
        sys.exit(1)

    print(f'预测序列数: {len(test_sequences)}', file=sys.stderr)
    predictions = predict_peptide_property(args.model_path, test_sequences)

    # 输出结果
    out = open(args.output, 'w') if args.output else sys.stdout
    for seq, pred in zip(test_sequences, predictions):
        true_val = f' {dic[seq]}' if seq in dic else ''
        out.write(f'序列 {seq}: 预测性质值 = {pred:.2f}{true_val}\n')
    if args.output:
        out.close()
        print(f'预测结果已保存到: {args.output}', file=sys.stderr)


def run_extrapolate(args):
    """外推模式：随机生成序列，筛选高性质值候选"""
    sequences, properties, dic = load_data(args.data)

    out = open(args.output, 'w') if args.output else sys.stdout

    print(time.time(), file=sys.stderr)
    for seq_len in range(args.seq_len_min, args.seq_len_max):
        ss = set()
        n = 0
        batch = []
        while len(ss) < args.target_count:
            seq = ''.join(random.choices(AMINO_ACIDS[:-1], k=seq_len))
            if seq in ss or seq in dic:
                continue
            n += 1
            batch.append(seq)
            if len(batch) >= min(625 * 2 ** (seq_len - 5), args.max_batch):
                predictions = predict_peptide_property(args.model_path, batch)
                for b, prob in zip(batch, predictions):
                    if prob > args.threshold:
                        ss.add(b)
                batch.clear()
        print(f'{time.time()} seq_len={seq_len} tried={n} found={len(ss)}', file=sys.stderr)
        for s in ss:
            out.write(s + '\n')

    if args.output:
        out.close()
        print(f'外推结果已保存到: {args.output}', file=sys.stderr)


if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'validate':
        run_validate(args)
    elif args.mode == 'predict':
        run_predict(args)
    elif args.mode == 'extrapolate':
        run_extrapolate(args)
