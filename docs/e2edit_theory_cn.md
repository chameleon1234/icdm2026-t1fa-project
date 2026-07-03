# E2EDiT 理论设计

E2EDiT 全称为 **End-to-End frequency-Decoupled Disease-informed Texture-preserving T1-to-FA Synthesis**，中文名为“端到端频率解耦疾病感知纹理保持 T1-to-FA 合成模型”。

## 1. Motivation

T1-to-FA 不是普通 image-to-image translation。单张 T1 能提供脑结构、组织边界和部分白质解剖线索，但并不完整包含真实 FA 的微观扩散纹理。因此模型如果只追求 PSNR/SSIM，容易退化为 posterior mean；如果只追求视觉清晰，又容易生成缺乏 FA 证据支持的假纹理。

此前路线的失败模式如下：

- **Posterior mean / MSE 路线**：指标稳定，但输出偏平滑，白质高频纹理被平均掉。
- **LPIPS / GAN 路线**：视觉更锐，但容易出现白质过亮、亮点、亮条和假纹理。
- **Direct T1-start MeanFlow / K25 路线**：高频强，但容易保留 T1-like texture，医学一致性不稳。
- **PriorFlow K8/K25**：存在 template / prior shortcut 风险，难以证明 subject-specific FA 高频来自当前 T1。
- **旧 FADiT / A080+DS**：A080 来自离线 base/detail prediction folders，detail 依赖 PriorFlow，因此只能作为工程参考或 upper bound，不能作为端到端主方法。
- **ReFA-Flow**：理论有意义，但 smoke no-go，wrong-T1 drop、residual_corr、PSNR/SSIM/WM/ROI 均未达到可用线。

## 2. Core hypothesis

可靠的 T1-to-FA 模型应当端到端地从当前 T1 中学习可支持的 FA 低频结构和高频细节，而不是依赖离线 prediction folder、template source 或 PriorFlow detail source。

英文假设：

> A reliable T1-to-FA synthesis model should decouple low-frequency FA structure and high-frequency FA detail within an end-to-end network, while using a bounded disease-informed corrector to improve medical consistency without hallucinating unsupported texture.

## 3. Model formulation

E2EDiT 分为两部分。

第一部分是端到端频率解耦 prior generator：

\[
B_i = B(T1_i)
\]

\[
H_i = H(T1_i)
\]

\[
P_i = clamp(B_i + H_i, -1, 1)
\]

其中：

- \(B_i\)：low-frequency FA base，只负责 FA 低频结构、整体强度空间、脑区大轮廓和白质/灰质大范围分布。
- \(H_i\)：high-frequency FA detail / residual，只负责局部边缘、白质纹理和 FA high-pass detail。
- \(P_i\)：Stage1 prior。

第二部分是 disease-informed bounded corrector：

\[
\Delta_i = C(T1_i, P_i, LP(P_i), HP(P_i), Edge(T1_i), WMproxy_i, Coord)
\]

\[
FA_i^{final} = clamp(P_i + \Delta_i, -1, 1)
\]

其中 \(\Delta_i\) 是小幅 bounded correction，不允许重新生成整张 FA。

## 4. Loss design

### M0: Single-branch baseline

普通 T1 -> FA 单分支模型，作为频率解耦是否有效的参照。

### M1: Dual-branch naive

模型输出 \(B(T1)\) 和 \(H(T1)\)，但只使用基础重建损失，检查双分支本身是否有帮助。

### M2: Dual-branch + anti-mean constraints

M2 是 E2EDiT Stage1 的核心验证版本：

- \(L_{base-low}\)：让 \(B(T1)\) 对齐 \(LP(FA)\)，限制 base 只学低频。
- \(L_{base-hf-suppress}\)：惩罚 \(HP(B)\)，防止 base 偷学完整 FA。
- \(L_{detail-hp}\)：让 \(HP(P)\) 对齐 \(HP(FA)\)，推动 \(H(T1)\) 承担高频。
- \(L_{recon}\)、\(L_{ssim}\)、\(L_{wm}\)、\(L_{roi}\)、\(L_{artifact}\)：保证重建、医学一致性和伪影控制。

### M3: M2 + bounded corrector

M3 在冻结或继承 M2 Stage1 的基础上训练 bounded corrector。Corrector 的目标不是补更多纹理，而是：

- 降低 WM-MAE。
- 提升 ROI-CCC proxy。
- 改善 PSNR/SSIM。
- 保持 SharpRatio 合理。
- 不抹掉 Stage1 有效纹理。

## 5. Smoke validation

E2EDiT 必须通过两个核心消融：

1. **频率解耦是否成立**：M2 必须相对 M0 更好或不明显更差，并且 SharpRatio、HF-Corr、B/H 分支诊断要支持“B 低频、H 高频”的角色分工。
2. **corrector 是否成立**：M3 必须相对 M2 提升 PSNR、WM-MAE、ROI proxy，同时不破坏清晰度。

如果 M2 不优于 M0，则频率解耦没有被证明。如果 M3 不优于 M2，则 corrector 没有被证明。任一失败都必须停止，不允许强行 full-data training。

## 6. What E2EDiT explicitly does not use

E2EDiT 第一版 smoke 严格不使用：

- A080 prediction folder。
- PriorFlow K8/K25。
- template source。
- 离线 base/detail folder。
- blend_manifest。
- GAN / LPIPS。
- disease label 作为推理输入。
- test set 调参。

## 7. Innovation points

E2EDiT 的可写创新点是：

1. **End-to-end frequency-decoupled prior generation**：在单一端到端网络内显式分离低频 FA base 和高频 FA detail。
2. **Anti-mean low/high branch constraints**：用低频监督和 base 高频抑制防止模型退化成 posterior mean 或让 base 偷学完整 FA。
3. **Disease-informed bounded correction**：Stage2 只做小幅医学一致性校正，不重新生成图像，也不破坏 Stage1 纹理。

这些创新必须由 M0/M2/M3 smoke 消融证明，而不是只靠最终指标包装。
