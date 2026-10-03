# DeepWeeds Day 2 — mã chạy lại

Mã bài làm nằm trong `code/`. Chạy các lệnh dưới đây từ **thư mục gốc repo** trên máy có GPU. Notebook `code/lab_day2.ipynb` gọi cùng các lệnh. Không sửa `eval.py` gốc. Các file CSV của tác giả dùng fold 0 cố định; test chỉ được đọc ở bước `final`.

## Cài đặt và thứ tự chạy

```bash
python -m pip install -r submissions/deepweeds_lab/code/requirements.txt
python submissions/deepweeds_lab/code/experiments.py prepare
python submissions/deepweeds_lab/code/experiments.py eda
python submissions/deepweeds_lab/code/experiments.py sanity
python submissions/deepweeds_lab/code/experiments.py backbones
python submissions/deepweeds_lab/code/experiments.py training
```

Kiểm tra CPU trước khi chạy GPU: `python -m unittest discover -s submissions/deepweeds_lab/code -p 'test_*.py' -v`. Bộ này kiểm Focal γ=0, CutMix, nhiệt độ, gộp BatchNorm, thứ tự ảnh và benchmark.

`prepare` tải `images.zip` từ [Zenodo](https://zenodo.org/records/7939060), tiếp tục phần tải dở bằng `curl -C -`, kiểm MD5 `b7b30f96d466fba86016aa5a26606e0f`, giải nén vào `data/images/`, và tải các CSV gốc. `eda` kiểm số ảnh, lớp, giao rỗng và sự hiện diện của ảnh trước huấn luyện. `sanity` kiểm Focal γ=0, overfit một batch và ảnh CutMix.

Lưu ý dữ liệu gốc: các file fold chỉ có `Filename,Label`. Một ảnh `20170714-110407-3.jpg` có nhãn 1 ở `labels.csv` nhưng nhãn 0 ở fold 0. Code giữ nguyên nhãn fold 0 để train/eval, đồng thời ghi sai khác vào `eda.json`.

`backbones` chạy B01–B05 (ResNet-50, ResNeXt-50, ConvNeXt-T, DeiT-S, EfficientNet-B0). `training` chạy T01–T11, mỗi lần thay một yếu tố của baseline B01, rồi T12 kết hợp augmentation và loss tốt nhất theo val. Baseline ở vòng cuối là `T00`, cùng cấu hình với B01. Tất cả đều dùng seed 0, fold 0, 12 epoch, batch 64, AdamW, warmup + cosine, chọn checkpoint theo macro-F1 val. Các ảnh đường cong được tạo trong `curves/`; checkpoint/log/config trong `runs/`.

**Sau khi xem `runs/*/seed0/summary.json`, chọn `SOURCE` bằng macro-F1 val và chi phí tính toán.** Script suy luận hiện hỗ trợ crop/scale với backbone CNN; nên chọn B01 hoặc một T01–T11 để có đủ phương pháp.

```bash
python submissions/deepweeds_lab/code/experiments.py inference --source T03
```

Xem `inference_results.json` để chọn `METHOD` từ I00–I05 theo macro-F1 val, ECE và p95. I01 là lật ngang, I02 là 5 crop, I03 là hai độ phân giải, I04 là temperature scaling khớp trên val, I05 là gộp Conv+BatchNorm. **Chỉ sau khi chốt SOURCE và METHOD trên val**, chạy:

```bash
python submissions/deepweeds_lab/code/experiments.py final --source T03 --method I04
python submissions/deepweeds_lab/code/experiments.py export
python eval.py score --pred 'submissions/deepweeds_lab/predictions/F01_seed*_test.csv' --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out
python eval.py score --pred 'submissions/deepweeds_lab/predictions/T00_seed*_test.csv' --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag T00 --out eval_out
python eval.py grade --final 'submissions/deepweeds_lab/predictions/F01_seed*_test.csv' --baseline 'submissions/deepweeds_lab/predictions/T00_seed*_test.csv' --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

`final` huấn luyện lại F01 và T00 với seed 0, 1, 2; đánh giá test đúng một lượt mỗi seed. Nếu chọn I04, `F01uncal_seed*_test.csv` được lưu từ cùng logit test để đối chiếu ECE, không đọc lại test. `export` tạo `results.xlsx` từ log và file dự đoán thật. Chỉ tạo báo cáo sau khi các thí nghiệm thực sự hoàn tất.

Phiên bản thư viện, seed, tag trọng số, cấu hình và thời gian epoch được ghi trong `runs/<exp_id>/seed<k>/config.json`, `history.csv`, `summary.json`. Tên notebook chạy lại là `code/lab_day2.ipynb`; đường dẫn notebook Colab/Kaggle công khai phải được thêm sau khi bạn tải notebook lên nền tảng đó.
