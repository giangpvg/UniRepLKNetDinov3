# UniRepLKNet with DINOv3 Backbone

Cập nhật **DINOv3 (Vision Transformer)** làm backbone thay thế cho phần trích xuất đặc trưng thuần tích chập (Large-Kernel CNN) của [UniRepLKNet](./UniRepLKNet/unireplknet.py).


> **Ghi chú:** Dự án được phát triển dựa trên việc kết hợp và tham khảo từ 2 repository mã nguồn mở:
> - **UniRepLKNet**: [AILab-CVC/UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet)
> - **DINOv3**: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

---

## Các điểm cập nhật

1. **Thay thế Backbone:**
   - Thay thế toàn bộ Stem và 4 Stage CNN lớn bằng DINOv3 ViT (`dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`) nạp trọng số tự giám sát (self-supervised) từ tập dữ liệu khổng lồ `LVD-1689M`.
   - Loại bỏ các thành phần phức tạp của CNN: không cần cấu hình kernel lớn, thư viện iGEMM hay bước tái tham số hóa (`reparameterize`).

2. **Giả lập Feature Pyramid (Đa tỷ lệ) cho Downstream Tasks:**
   - ViT mặc định có độ phân giải cố định (stride 16).
   - Hệ thống trích xuất 4 tầng trung gian cách đều nhau và tái tạo lại 4 tỷ lệ phân giải tương thích với downstream heads (FPN / UPerNet):
     - `f1`: Bilinear interpolation $\times 4$ $\rightarrow$ **Stride 4**
     - `f2`: Bilinear interpolation $\times 2$ $\rightarrow$ **Stride 8**
     - `f3`: Giữ nguyên $\rightarrow$ **Stride 16**
     - `f4`: Max pooling $2\times2$ (stride 2) $\rightarrow$ **Stride 32**

3. **Hỗ trợ số kênh đầu vào tùy biến (`in_chans`):**
   - Hỗ trợ dữ liệu có số kênh khác 3 (ví dụ: `in_chans=1` cho Audio Spectrogram).
   - Tự động gộp và tính trung bình trọng số pretrained RGB sang số kênh mới, giúp tận dụng trọng số pretrain mà không cần train lại patch projection từ đầu.

4. **Tối ưu hóa Phân loại (Global Pooling):**
   - Bổ sung hàm `forward_global()` lấy trung bình trực tiếp từ các patch tokens đã chuẩn hóa (`x_norm_patchtokens.mean(dim=1)`), bỏ qua bước nội suy pyramid khi chạy bài toán phân loại (`output_mode='logits'`).

5. **Stochastic Depth (Drop Path):**
   - Tự động cấu hình tỉ lệ Drop Path tăng dần tuyến tính qua các khối transformer nhằm chống overfitting khi fine-tune.

6. **Tương thích OpenMMLab / MMSegmentation:**
   - Giữ nguyên toàn bộ giao diện: nạp checkpoint `init_cfg`, 2 chế độ đầu ra (`logits` và `features`), đăng ký sẵn `@seg_BACKBONES.register_module()`.

---

## Tham khảo (References & Acknowledgements)

Mã nguồn được kế thừa và tham khảo từ 2 dự án mã nguồn mở:
1. **[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet)** (*A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition*): Kế thừa cấu trúc head phân loại, lớp chuẩn hóa LayerNorm, các output mode (`logits`/`features`) và thiết lập tích hợp OpenMMLab/MMSegmentation.
2. **[DINOv3](https://github.com/facebookresearch/dinov3)** (*Meta AI Research, FAIR*): Sử dụng kiến trúc Vision Transformer (ViT) và trọng số tiền huấn luyện tự giám sát (tập `LVD-1689M`) làm backbone trích xuất đặc trưng nền tảng.
