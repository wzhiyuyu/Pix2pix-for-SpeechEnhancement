import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_STOI, compute_PESQ
from utils.utils import cal_lps, phase, lps_to_mag, rebuild_waveform, set_requires_grad, ExecutionTime, cal_mel, \
    mel_to_mag, \
    cal_log_mel, log_mel_to_mag, cal_mag

class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            G,
            D,
            optim_G,
            optim_D,
            loss_function,
            train_dl,
            validation_dl,
    ):
        super(Trainer, self).__init__(config, resume, G, D, optim_G, optim_D, loss_function)
        self.train_data_loader = train_dl
        self.validation_data_loader = validation_dl

    def _set_model_train(self):
        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/13
        self.generator.train()
        self.discriminator.train()


    def _set_model_eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def _transform_pesq_range(self, pesq_score):
        """平移 PESQ 评价指标
        将 PESQ 评价指标的范围从 -0.5 ~ 4.5 平移为 0 ~ 1
        Args:
            pesq_score: PESQ 得分
        Returns:
            0 ~ 1 范围的 PESQ 得分
        """

        return (pesq_score + 0.5) * 2 / 10


    def _is_best_score(self, score):
        """检查当前的结果是否为最佳模型"""
        if score >= self.best_score:
            self.best_score = score
            return True
        else:
            return False


    def _visualize_weights_and_grads(self, model, epoch):
        for name, param in model.named_parameters():
            self.viz.writer.add_histogram("WEIGHT_" + name, param.clone().cpu().data.numpy(), epoch)
            # param.clone() 仅仅拷贝 param.data
            # [TODO 验证] 拷贝到 cpu ，param.cpu() 返回值中是否包含梯度
            self.viz.writer.add_histogram("GRAD_" + name, param.grad.cpu().numpy(), epoch)

    def _train_epoch(self, epoch):
        """定义单次训练"""
        self._set_model_train()

        # https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510/4
        # The length of the loader will adapt to the batch_size
        batch_size = self.train_data_loader.batch_size
        n_batch = len(self.train_data_loader)

        visualize_loss = lambda tag, loss, itr: self.viz.writer.add_scalar(f"损失/{tag}", loss, itr)
        visualize_pred = lambda tag, pred, itr: self.viz.writer.add_scalar(f"判别结果/{tag}", pred, itr)

        for i, (real_A, real_B, basename_text) in enumerate(self.train_data_loader):
            # 之前进行的 epoch 数量 * 每个 epoch 包含的 batch 数量 * batch_size + 当前为第 i 个 batch
            iteration = (epoch - 1) * n_batch * batch_size + i * batch_size

            real_A = real_A.to(self.dev)
            real_B = real_B.to(self.dev)

            fake_B = self.generator(real_A)

            if self.soft_label:
                real_label = np.random.uniform(0.7, 1.2)
                fake_label = 0.0
            else:
                real_label = 1.0
                fake_label = 0.0

            """# ============ 更新 D 网络 ============ #"""
            if i % self.iter_discriminator_period == 0:
                set_requires_grad(self.discriminator, True)
                self.optimizer_D.zero_grad()

                # 假为假
                fake_AB = torch.cat((real_A, fake_B), dim=1)
                pred_fake_in_optim_D = self.discriminator(fake_AB.detach())
                loss_D_fake = self.loss_gan(pred_fake_in_optim_D,
                                            torch.full(pred_fake_in_optim_D.shape, fake_label).to(self.dev))
                # 真为真
                real_AB = torch.cat((real_A, real_B), dim=1)
                pred_real_in_optim_D = self.discriminator(real_AB)
                loss_D_real = self.loss_gan(pred_real_in_optim_D,
                                            torch.full(pred_real_in_optim_D.shape, real_label).to(self.dev))
                # 组合
                loss_D = (loss_D_fake + loss_D_real) * 0.5
                # 更新参数
                loss_D.backward()
                self.optimizer_D.step()

                with torch.no_grad():
                    visualize_loss("D", loss_D, iteration)
                    visualize_loss("D_real", loss_D_real, iteration)
                    visualize_loss("D_fake", loss_D_fake, iteration)

                    visualize_pred("pred_real_in_optim_D", torch.mean(pred_real_in_optim_D), iteration)
                    visualize_pred("pred_fake_in_optim_D", torch.mean(pred_fake_in_optim_D), iteration)

            # self._visualize_weights_and_grads(self.discriminator, epoch)

            """# ============ 更新 G 网络 ============ #"""
            set_requires_grad(self.discriminator, False)
            self.optimizer_G.zero_grad()

            # 假为真
            fake_AB = torch.cat((real_A, fake_B), dim=1)
            pred_fake_in_optim_G = self.discriminator(fake_AB)
            loss_G_fake = self.loss_gan(pred_fake_in_optim_G, torch.ones(pred_fake_in_optim_G.shape).to(self.dev))
            loss_G_fake *= self.adversarial_loss_factor
            # 额外的损失，帮助收敛
            loss_G_addition = self.loss_addition(fake_B, real_B)
            loss_G_addition *= self.additional_loss_factor
            # 组合
            loss_G = loss_G_fake + loss_G_addition
            loss_G.backward()
            self.optimizer_G.step()

            # self._visualize_weights_and_grads(self.generator, epoch)

            with torch.no_grad():
                visualize_loss("G", loss_G, iteration)
                visualize_loss("G_fake", loss_G_fake, iteration)
                visualize_loss("G_addition", loss_G_addition, iteration)
                visualize_pred("pred_fake_in_optim_G", torch.mean(pred_fake_in_optim_G), iteration)

    @torch.no_grad()
    def enhancement_and_visualization(self, data, epoch):
        if self.feature == "mag":
            return self.enhance_and_visualize_mag(*data, epoch)
        elif self.feature == "lps":
            return self.enhance_and_visualize_lps(*data, epoch)
        elif self.feature == "mel":
            return self.enhance_and_visualize_mel(*data, epoch)
        elif self.feature == "log_mel":
            return self.enhance_and_visualize_log_mel(*data, epoch)
        else:
            raise ValueError

    @torch.no_grad()
    def enhance_and_visualize_mag(self, mixture, clean, mixture_name, norm_meta, epoch):
        """
        Notes:
            1. mixture => LPS, Phase
            2. 去掉最后一帧，最后 1 bin
            3. 分割为 256 长度的子 LPS（不补全最后一个，直接丢弃）
            5. 根据 norm_meta 判断是否需要计算 max 与 min 特征值，之后对各个 子 LPS 进行规范化
            6. 通过模型增强语音
            7. 进行反规划化
            8. 拼接增强之后的结果
            9. 使用第 2 步提取的最后一个 bin 来补全增强后的结果
            10. 计算纯净语音的 LPS，并整理维度，可视化 mixture，clean，enhanced，mixture_name
            11. 结合带噪语音的相位，将 LPS 还原为时域信号
            12. 矫正长度
            13. 返回矫正后的 mixture，clean，enhanced，mixture_name
        """
        mixture_sp = cal_mag(mixture)[:, :-1]  # 抛弃最后一帧
        mixture_phase = phase(mixture)[:, :-1]  # 抛弃最后一帧
        last_bin_sp = mixture_sp[-1, :]
        mixture_sp = mixture_sp[:-1, :]  # 原尺寸为 (257, T)，为了方便计算，抛弃最后一个 bin

        # 开始张量运算
        mixture_chunks = torch.split(torch.Tensor(mixture_sp), 256, dim=1)
        if len(mixture_chunks) > 1:
            # 数据集保证一定可以提取出至少一个 t=256 的 LPS 子频谱
            # 但当 t 正好等于 256 时，会出问题：e.g. a = [1], a[:-1] == []
            mixture_chunks = mixture_chunks[:-1]  # Del last one

        enhanced_chunks = []
        for j, chunk in enumerate(mixture_chunks):
            norm_max = 0.0
            norm_min = 0.0
            if norm_meta["type"] in (1, 2, 3):
                if norm_meta["type"] == 1:
                    norm_max = norm_meta["eigenvalues"]["max"]
                    norm_min = norm_meta["eigenvalues"]["min"]
                elif norm_meta["type"] == 2:
                    norm_max = torch.max(chunk)
                    norm_min = torch.min(chunk)
                else:
                    norm_max = torch.max(chunk, 0)
                    norm_min = torch.min(chunk, 0)
                chunk = 2 * (chunk - norm_min) / (norm_max - norm_min) - 1
                # norm_meta == 0 时，不进行任何规范化

            chunk = chunk.reshape(1, 1, 256, 256)
            enhanced_chunk = self.generator(chunk).squeeze()  # 256, 256
            if norm_meta["type"] in (1, 2, 3):
                enhanced_chunk = (enhanced_chunk + 1) * (norm_max - norm_min) / 2 + norm_min
            enhanced_chunks.append(enhanced_chunk.detach().numpy())

        enhanced_sp = np.concatenate(enhanced_chunks, axis=1)

        # 带噪语音的 last_bin 被用在增强后的语音中，但需要先删除额外的长度
        last_bin_sp = last_bin_sp[:enhanced_sp.shape[1]].reshape(1, -1)
        enhanced_sp = np.concatenate([enhanced_sp, last_bin_sp], axis=0)  # Pad with last lps bin
        assert enhanced_sp.shape[0] == 257

        # Spectrum 的可视化
        fig, ax = plt.subplots(3, 1)
        for j, sp in enumerate([
            mixture_sp[:, :enhanced_sp.shape[1]],
            enhanced_sp,
            cal_mag(clean)[:, :enhanced_sp.shape[1]]
        ]):
            ax[j].set_title(
                f"mean: {round(float(np.mean(sp)), 3)} "
                f"std: {round(float(np.std(sp)), 3)} "
                f"max: {round(float(np.max(sp)), 3)} "
                f"min: {round(float(np.min(sp)), 3)}"
            )
            ax[j].imshow(sp, interpolation="nearest", origin="lower")
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        fig.subplots_adjust(hspace=0.5)

        self.viz.writer.add_figure(f"语音频谱/{mixture_name}", fig, epoch)

        # Rebuild wav
        enhanced_mag = enhanced_sp
        enhanced = rebuild_waveform(enhanced_mag, mixture_phase[:, :enhanced_mag.shape[1]])

        min_length = min(len(mixture), len(enhanced), len(clean))
        mixture = mixture[:min_length]
        enhanced = enhanced[:min_length]
        clean = clean[:min_length]

        fig, ax = plt.subplots(3, 1)
        for j, y in enumerate([mixture, enhanced, clean]):
            ax[j].set_title(
                f"mean: {np.mean(y):3f}, std: {np.std(y):3f}, max: {np.max(y):3f}, min: {np.min(y):3f}")
            librosa.display.waveplot(y, sr=16000, ax=ax[j])
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        fig.subplots_adjust(hspace=0.5)

        self.viz.writer.add_figure(f"语音波形图像/{mixture_name}", fig, epoch)

        return (
            compute_STOI(clean, mixture, sr=16000),
            compute_STOI(clean, enhanced, sr=16000),
            compute_PESQ(clean, mixture, sr=16000),
            compute_PESQ(clean, enhanced, sr=16000)
        )

    @torch.no_grad()
    def enhance_and_visualize_log_mel(self, mixture, clean, mixture_name, norm_meta, epoch):
        """
        Notes:
            1. mixture => Mag, Phase
            3. 分割为 256 长度的子 Mel（不补全最后一个，直接丢弃）
            5. 根据 norm_meta 判断是否需要计算 max 与 min 特征值，之后对各个 子 Mel 进行规范化
            6. 通过模型增强语音
            7. 进行反规划化
            8. 拼接增强之后的结果
            10. 计算纯净语音，并整理维度，可视化 mixture，clean，enhanced，mixture_name
            11. 结合带噪语音的相位，还原为时域信号
            12. 矫正长度
            13. 返回矫正后的 mixture，clean，enhanced，mixture_name
        """
        mixture_mel = cal_log_mel(mixture)
        mixture_phase = phase(mixture, n_fft=1024)  # (257, T)

        # 开始张量运算
        mixture_chunks = torch.split(torch.Tensor(mixture_mel), 256, dim=1)
        if len(mixture_chunks) > 1:
            # 数据集保证一定可以提取出至少一个 t=256 的 LPS 子频谱
            # 但当 t 正好等于 256 时，会出问题：e.g. a = [1], a[:-1] == []
            mixture_chunks = mixture_chunks[:-1]  # Del last one

        enhanced_chunks = []
        for j, chunk in enumerate(mixture_chunks):
            norm_max = 0.0
            norm_min = 0.0
            if norm_meta["type"] in (1, 2, 3):
                if norm_meta["type"] == 1:
                    norm_max = norm_meta["eigenvalues"]["max"]
                    norm_min = norm_meta["eigenvalues"]["min"]
                elif norm_meta["type"] == 2:
                    norm_max = torch.max(chunk)
                    norm_min = torch.min(chunk)
                else:
                    norm_max = torch.max(chunk, 0)
                    norm_min = torch.min(chunk, 0)
                chunk = 2 * (chunk - norm_min) / (norm_max - norm_min) - 1
                # norm_meta == 0 时，不进行任何规范化

            chunk = chunk.reshape(1, 1, 256, 256)
            enhanced_chunk = self.generator(chunk).squeeze()  # 256, 256
            if norm_meta["type"] in (1, 2, 3):
                enhanced_chunk = (enhanced_chunk + 1) * (norm_max - norm_min) / 2 + norm_min
            enhanced_chunks.append(enhanced_chunk.detach().numpy())

        enhanced_mel = np.concatenate(enhanced_chunks, axis=1)
        assert enhanced_mel.shape[0] == 256  # (256, T)

        # Spectrum 的可视化
        fig, ax = plt.subplots(3, 1)
        for j, sp in enumerate([
            mixture_mel[:, :enhanced_mel.shape[1]],
            enhanced_mel,
            cal_log_mel(clean)[:, :enhanced_mel.shape[1]]
        ]):
            ax[j].set_title(
                f"mean: {round(float(np.mean(sp)), 3)} "
                f"std: {round(float(np.std(sp)), 3)} "
                f"max: {round(float(np.max(sp)), 3)} "
                f"min: {round(float(np.min(sp)), 3)}"
            )
            ax[j].imshow(sp, interpolation="nearest", origin="lower")
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        fig.subplots_adjust(hspace=0.5)

        self.viz.writer.add_figure(f"语音频谱/{mixture_name}", fig, epoch)

        # Rebuild wav
        enhanced_mag = log_mel_to_mag(enhanced_mel)
        enhanced = rebuild_waveform(enhanced_mag, mixture_phase[:, :enhanced_mag.shape[1]])

        min_length = min(len(mixture), len(enhanced), len(clean))
        mixture = mixture[:min_length]
        enhanced = enhanced[:min_length]
        clean = clean[:min_length]

        fig, ax = plt.subplots(3, 1)
        for j, y in enumerate([mixture, enhanced, clean]):
            ax[j].set_title(
                f"mean: {np.mean(y):3f}, std: {np.std(y):3f}, max: {np.max(y):3f}, min: {np.min(y):3f}")
            librosa.display.waveplot(y, sr=16000, ax=ax[j])
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        fig.subplots_adjust(hspace=0.5)

        self.viz.writer.add_figure(f"语音波形图像/{mixture_name}", fig, epoch)

        return (
            compute_STOI(clean, mixture, sr=16000),
            compute_STOI(clean, enhanced, sr=16000),
            compute_PESQ(clean, mixture, sr=16000),
            compute_PESQ(clean, enhanced, sr=16000)
        )

    @torch.no_grad()
    def enhance_and_visualize_mel(self, mixture, clean, mixture_name, norm_meta, epoch):
        """
        Notes:
            1. mixture => Mag, Phase
            3. 分割为 256 长度的子 Mel（不补全最后一个，直接丢弃）
            5. 根据 norm_meta 判断是否需要计算 max 与 min 特征值，之后对各个 子 Mel 进行规范化
            6. 通过模型增强语音
            7. 进行反规划化
            8. 拼接增强之后的结果
            10. 计算纯净语音，并整理维度，可视化 mixture，clean，enhanced，mixture_name
            11. 结合带噪语音的相位，还原为时域信号
            12. 矫正长度
            13. 返回矫正后的 mixture，clean，enhanced，mixture_name
        """
        mixture_mel = cal_mel(mixture)
        mixture_phase = phase(mixture, n_fft=1024)  # (257, T)

        # 开始张量运算
        mixture_chunks = torch.split(torch.Tensor(mixture_mel), 256, dim=1)
        if len(mixture_chunks) > 1:
            # 数据集保证一定可以提取出至少一个 t=256 的 LPS 子频谱
            # 但当 t 正好等于 256 时，会出问题：e.g. a = [1], a[:-1] == []
            mixture_chunks = mixture_chunks[:-1]  # Del last one

        enhanced_chunks = []
        for j, chunk in enumerate(mixture_chunks):
            norm_max = 0.0
            norm_min = 0.0
            if norm_meta["type"] in (1, 2, 3):
                if norm_meta["type"] == 1:
                    norm_max = norm_meta["eigenvalues"]["max"]
                    norm_min = norm_meta["eigenvalues"]["min"]
                elif norm_meta["type"] == 2:
                    norm_max = torch.max(chunk)
                    norm_min = torch.min(chunk)
                else:
                    norm_max = torch.max(chunk, 0)
                    norm_min = torch.min(chunk, 0)
                chunk = 2 * (chunk - norm_min) / (norm_max - norm_min) - 1
                # norm_meta == 0 时，不进行任何规范化

            chunk = chunk.reshape(1, 1, 256, 256)
            enhanced_chunk = self.generator(chunk).squeeze()  # 256, 256
            if norm_meta["type"] in (1, 2, 3):
                enhanced_chunk = (enhanced_chunk + 1) * (norm_max - norm_min) / 2 + norm_min
            enhanced_chunks.append(enhanced_chunk.detach().numpy())

        enhanced_mel = np.concatenate(enhanced_chunks, axis=1)
        assert enhanced_mel.shape[0] == 256  # (256, T)

        # Spectrum 的可视化
        fig, ax = plt.subplots(3, 1)
        for j, sp in enumerate([
            mixture_mel[:, :enhanced_mel.shape[1]],
            enhanced_mel,
            cal_mel(clean)[:, :enhanced_mel.shape[1]]
        ]):
            ax[j].set_title(
                f"mean: {round(float(np.mean(sp)), 3)} "
                f"std: {round(float(np.std(sp)), 3)} "
                f"max: {round(float(np.max(sp)), 3)} "
                f"min: {round(float(np.min(sp)), 3)}"
            )
            ax[j].imshow(sp, interpolation="nearest", origin="lower")
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        fig.subplots_adjust(hspace=0.5)

        self.viz.writer.add_figure(f"语音频谱/{mixture_name}", fig, epoch)

        # Rebuild wav
        enhanced_mag = mel_to_mag(enhanced_mel)
        enhanced = rebuild_waveform(enhanced_mag, mixture_phase[:, :enhanced_mag.shape[1]])

        min_length = min(len(mixture), len(enhanced), len(clean))
        mixture = mixture[:min_length]
        enhanced = enhanced[:min_length]
        clean = clean[:min_length]

        fig, ax = plt.subplots(3, 1)
        for j, y in enumerate([mixture, enhanced, clean]):
            ax[j].set_title(
                f"mean: {np.mean(y):3f}, std: {np.std(y):3f}, max: {np.max(y):3f}, min: {np.min(y):3f}")
            librosa.display.waveplot(y, sr=16000, ax=ax[j])
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        fig.subplots_adjust(hspace=0.5)

        self.viz.writer.add_figure(f"语音波形图像/{mixture_name}", fig, epoch)

        return (
            compute_STOI(clean, mixture, sr=16000),
            compute_STOI(clean, enhanced, sr=16000),
            compute_PESQ(clean, mixture, sr=16000),
            compute_PESQ(clean, enhanced, sr=16000)
        )

    @torch.no_grad()
    def enhance_and_visualize_lps(self, mixture, clean, mixture_name, norm_meta, epoch):
        """
        Notes:
            1. mixture => LPS, Phase
            2. 去掉最后一帧，最后 1 bin
            3. 分割为 256 长度的子 LPS（不补全最后一个，直接丢弃）
            5. 根据 norm_meta 判断是否需要计算 max 与 min 特征值，之后对各个 子 LPS 进行规范化
            6. 通过模型增强语音
            7. 进行反规划化
            8. 拼接增强之后的结果
            9. 使用第 2 步提取的最后一个 bin 来补全增强后的结果
            10. 计算纯净语音的 LPS，并整理维度，可视化 mixture，clean，enhanced，mixture_name
            11. 结合带噪语音的相位，将 LPS 还原为时域信号
            12. 矫正长度
            13. 返回矫正后的 mixture，clean，enhanced，mixture_name
        """
        mixture_lps = cal_lps(mixture)[:, :-1]  # 抛弃最后一帧
        mixture_phase = phase(mixture)[:, :-1]  # 抛弃最后一帧
        last_bin_lps = mixture_lps[-1, :]
        mixture_lps = mixture_lps[:-1, :]  # 原尺寸为 (257, T)，为了方便计算，抛弃最后一个 bin

        # 开始张量运算
        mixture_chunks = torch.split(torch.Tensor(mixture_lps), 256, dim=1)
        if len(mixture_chunks) > 1:
            # 数据集保证一定可以提取出至少一个 t=256 的 LPS 子频谱
            # 但当 t 正好等于 256 时，会出问题：e.g. a = [1], a[:-1] == []
            mixture_chunks = mixture_chunks[:-1]  # Del last one

        enhanced_chunks = []
        for j, chunk in enumerate(mixture_chunks):
            norm_max = 0.0
            norm_min = 0.0
            if norm_meta["type"] in (1, 2, 3):
                if norm_meta["type"] == 1:
                    norm_max = norm_meta["eigenvalues"]["max"]
                    norm_min = norm_meta["eigenvalues"]["min"]
                elif norm_meta["type"] == 2:
                    norm_max = torch.max(chunk)
                    norm_min = torch.min(chunk)
                else:
                    norm_max = torch.max(chunk, 0)
                    norm_min = torch.min(chunk, 0)
                chunk = 2 * (chunk - norm_min) / (norm_max - norm_min) - 1
                # norm_meta == 0 时，不进行任何规范化

            chunk = chunk.reshape(1, 1, 256, 256)
            enhanced_chunk = self.generator(chunk).squeeze()  # 256, 256
            if norm_meta["type"] in (1, 2, 3):
                enhanced_chunk = (enhanced_chunk + 1) * (norm_max - norm_min) / 2 + norm_min
            enhanced_chunks.append(enhanced_chunk.detach().numpy())

        enhanced_lps = np.concatenate(enhanced_chunks, axis=1)
        # 带噪语音的 last_bin 被用在增强后的语音中，但需要先删除额外的长度
        last_bin_lps = last_bin_lps[:enhanced_lps.shape[1]].reshape(1, -1)
        enhanced_lps = np.concatenate([enhanced_lps, last_bin_lps], axis=0)  # Pad with last lps bin
        assert enhanced_lps.shape[0] == 257

        # Spectrum 的可视化
        fig, ax = plt.subplots(3, 1)
        for j, sp in enumerate([
            mixture_lps[:, :enhanced_lps.shape[1]],
            enhanced_lps,
            cal_lps(clean)[:, :enhanced_lps.shape[1]]
        ]):
            ax[j].set_title(
                f"mean: {round(float(np.mean(sp)), 3)} "
                f"std: {round(float(np.std(sp)), 3)} "
                f"max: {round(float(np.max(sp)), 3)} "
                f"min: {round(float(np.min(sp)), 3)}"
            )
            ax[j].imshow(sp, interpolation="nearest", origin="lower")
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        fig.subplots_adjust(hspace=0.5)

        self.viz.writer.add_figure(f"语音频谱/{mixture_name}", fig, epoch)

        # Rebuild wav
        enhanced_mag = lps_to_mag(enhanced_lps)
        enhanced = rebuild_waveform(enhanced_mag, mixture_phase[:, :enhanced_mag.shape[1]])

        min_length = min(len(mixture), len(enhanced), len(clean))
        mixture = mixture[:min_length]
        enhanced = enhanced[:min_length]
        clean = clean[:min_length]

        fig, ax = plt.subplots(3, 1)
        for j, y in enumerate([mixture, enhanced, clean]):
            ax[j].set_title(
                f"mean: {np.mean(y):3f}, std: {np.std(y):3f}, max: {np.max(y):3f}, min: {np.min(y):3f}")
            librosa.display.waveplot(y, sr=16000, ax=ax[j])
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        fig.subplots_adjust(hspace=0.5)

        self.viz.writer.add_figure(f"语音波形图像/{mixture_name}", fig, epoch)

        return (
            compute_STOI(clean, mixture, sr=16000),
            compute_STOI(clean, enhanced, sr=16000),
            compute_PESQ(clean, mixture, sr=16000),
            compute_PESQ(clean, enhanced, sr=16000)
        )

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """测试逻辑
        测试时使用测试集，为了后续并行计算，首先需要将模型迁移至 CPU 上
        """
        self._set_model_eval()
        self.generator.to("cpu")

        stoi_clean_mixture = []
        stoi_clean_enhanced = []
        pesq_clean_mixture = []
        pesq_clean_enhanced = []
        for i, data in enumerate(self.validation_data_loader):
            # 每个内核计算的结果会被汇总在一个 list 中
            # [(return_1,return_2,return_3,return_4), ..., ]
            for j in range(self.validation_data_loader.batch_size):
                metrics = self.enhancement_and_visualization(data[j], epoch)
                stoi_clean_mixture.append(metrics[0])
                stoi_clean_enhanced.append(metrics[1])
                pesq_clean_mixture.append(metrics[2])
                pesq_clean_enhanced.append(metrics[3])

            # metrics = Parallel(n_jobs=40, require="sharedmem")(delayed(self.enhancement_and_visualization)(
            #     *data[i],
            #     epoch
            # ) for i in range(self.validation_data_loader.batch_size))
            # for j in metrics:
            #     stoi_clean_mixture.append(j[0])
            #     stoi_clean_enhanced.append(j[1])
            #     pesq_clean_mixture.append(j[2])
            #     pesq_clean_enhanced.append(j[3])

        metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"评价指标均值/STOI", {
            "clean 与 mixture": metrics_ave(stoi_clean_mixture),
            "clean 与 enhanced": metrics_ave(stoi_clean_enhanced)
        }, epoch)
        self.viz.writer.add_scalars(f"评价指标均值/PESQ", {
            "clean 与 mixture": metrics_ave(pesq_clean_mixture),
            "clean 与 enhanced": metrics_ave(pesq_clean_enhanced)
        }, epoch)

        score = (metrics_ave(stoi_clean_enhanced) + self._transform_pesq_range(metrics_ave(pesq_clean_enhanced))) / 2

        self.generator.to(self.dev)
        return score


    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============ Train epoch = {epoch} ============")
            print("[0 seconds] 开始训练...")
            timer = ExecutionTime()
            self.viz.set_epoch(epoch)

            self._train_epoch(epoch)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            if self.validation_period != 0 and epoch % self.validation_period == 0:
                # 测试一轮，并绘制波形文件
                print(f"[{timer.duration()} seconds] 训练结束，开始计算评价指标...")
                score = self._valid_epoch(epoch)

                if self._is_best_score(score):
                    self._save_checkpoint(epoch, is_best=True)

            print(f"[{timer.duration()} seconds] 完成当前 Epoch.")
