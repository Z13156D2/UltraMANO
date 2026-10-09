import os
import pytorch_lightning as pl
import json
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split
from pose_modules import UltrasoundToHandModule
from data import UltrasoundSingleFrameDataset, UltrasoundManoDataset
import transforms
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("lightning_logs", name="720_1_test")


def train_ultrasound2hand(data_dir, batch_size=16, max_epochs=100, num_workers=4):
    # 创建数据集
    dataset = UltrasoundManoDataset(
        data_dir=data_dir,
        transform=transforms.Compose([
            transforms.UltrasoundNormalize(mean=0.0,
                                           std=300.0, min_val=-1500, max_val=1500),
            # transforms.Signed_log_transforme(),
            transforms.RandomChannelShiftZeroPad(),
            transforms.AngleNormalize(),
            transforms.UltrasoundToTensor(),
        ])
    )

    # 分割数据集为训练集、验证集和测试集（70%/15%/15%）
    train_size = int(0.8 * len(dataset))
    val_size = int(0.19 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # 设置随机种子以确保可复现性
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 保存测试集文件列表
    test_indices = test_dataset.indices
    test_files = [dataset.file_paths[i] for i in test_indices]

    # 创建保存测试集文件列表的目录
    save_dir = "../test_file_list/"
    os.makedirs(save_dir, exist_ok=True)

    # 保存测试集文件列表到JSON文件
    test_files_path = os.path.join(save_dir, "test_files_720_3_test.json")
    with open(test_files_path, 'w') as f:
        json.dump(test_files, f, indent=4)

    print(f"测试集文件列表已保存到: {test_files_path}")
    print(f"测试集样本数量: {len(test_files)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 创建模型
    model = UltrasoundToHandModule(learning_rate=1e-4, weight_decay=1e-5)

    # 定义回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="/media/ubuntu/DATE/zzt/emg2pose/emg2pose/save_model_720_3_test",
        monitor="val_loss",
        filename="ultrasound2hand-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        verbose=True,
        mode="min",
    )

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",  # 自动选择GPU或CPU
        devices=1,  # 使用1个设备
        log_every_n_steps=10,
        logger=logger
    )

    # 训练模型
    trainer.fit(model, train_loader, val_loader)

    # 在测试集上评估模型
    # test_result = trainer.test(model, test_loader)
    # print(f"测试集结果: {test_result}")

    # 返回最佳模型路径
    return checkpoint_callback.best_model_path, test_files_path


if __name__ == "__main__":
    best_model_path, test_files_path = train_ultrasound2hand(
        data_dir="/media/ubuntu/DATE/zzt/emg2pose/emg2pose/ultradataset_720",
        batch_size=16,
        max_epochs=100
    )
    print(f"最佳模型保存在: {best_model_path}")
    print(f"测试集文件列表保存在: {test_files_path}")


