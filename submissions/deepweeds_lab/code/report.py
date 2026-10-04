"""Create the submission report only from completed runs and verified predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import BACKBONES, TRAINING, LABELS, PREDS, RUNS, SUB, CURVES
from eval import CLASS_NAMES, check_against_csv, compute_metrics, fmt, load_group, read_pred


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|",
                      *("| " + " | ".join(str(x) for x in row) + " |" for row in rows)])


def _summary(exp):
    path = RUNS / exp / "seed0" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def generate_report():
    import matplotlib.pyplot as plt
    from PIL import Image

    expected = {0, 1, 2}
    groups = {}
    for exp in ("T00", "F01"):
        group = load_group(str(PREDS / f"{exp}_seed*_test.csv"), str(LABELS / "test_subset0.csv"))
        if set(group.seeds) != expected:
            raise ValueError(f"{exp} requires exactly seeds 0, 1, 2; found {group.seeds}")
        for pred in group.preds:
            val = read_pred(str(PREDS / f"{exp}_seed{pred.seed}_val.csv"))
            check_against_csv(val, str(LABELS / "val_subset0.csv"), "val")
        groups[exp] = group
    if not (SUB / "results.xlsx").exists():
        raise FileNotFoundError("Run export to create results.xlsx first")
    baseline, final = groups["T00"], groups["F01"]
    cfg = json.loads((RUNS / "F01" / "seed0" / "config.json").read_text())
    baseline_cfg = json.loads((RUNS / "T00" / "seed0" / "config.json").read_text())
    eda = json.loads((SUB / "eda.json").read_text())
    sanity_path=SUB/"sanity.json"
    sanity=json.loads(sanity_path.read_text()) if sanity_path.exists() else None
    backbone_latency_path=SUB/"backbone_latency.json"
    backbone_latency=json.loads(backbone_latency_path.read_text()) if backbone_latency_path.exists() else {}
    gpu_name=backbone_latency.get("B01",{}).get("gpu",cfg.get("device","unknown"))
    infer = json.loads((SUB / "inference_results.json").read_text())
    locked = json.loads((SUB / "final_selection.json").read_text())
    if locked["method"] != cfg["final_inference"]:
        raise ValueError("Final inference differs from the pre-test selection")
    selected = next(r for r in infer if r["method"] == locked["method"] and r["source"] == locked["source"])
    if selected["source"] not in ("B01", "B02", "B03", "B04", "B05", *(f"T{i:02d}" for i in range(1, 13))):
        raise ValueError("Unknown validation-selected source")

    CURVES.mkdir(parents=True, exist_ok=True)
    # The confusion matrix is computed from one stated seed, never by mixing seed predictions.
    pred = next(p for p in final.preds if p.seed == 0)
    metric = next(m for p, m in zip(final.preds, final.metrics) if p.seed == 0)
    cm = metric["confusion"]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(9), CLASS_NAMES, rotation=55, ha="right")
    ax.set_yticks(range(9), CLASS_NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title="F01 seed 0: test confusion matrix")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CURVES / "F01_confusion_seed0.png", dpi=160)
    plt.close(fig)

    errors = np.flatnonzero(pred.y_true != pred.y_pred)
    errors = sorted(errors, key=lambda i: float(pred.probs[i, pred.y_pred[i]]), reverse=True)[:12]
    if errors:
        fig, axes = plt.subplots(3, 4, figsize=(14, 11))
        for ax, i in zip(axes.flat, errors):
            with Image.open(SUB.parents[1] / "data" / "images" / str(pred.filenames[i])) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"true: {CLASS_NAMES[pred.y_true[i]]}\npred: {CLASS_NAMES[pred.y_pred[i]]} ({pred.probs[i,pred.y_pred[i]]:.2f})", fontsize=9)
            ax.axis("off")
        for ax in list(axes.flat)[len(errors):]:
            ax.axis("off")
        fig.suptitle("F01 seed 0: highest-confidence test errors")
        fig.tight_layout()
        fig.savefig(CURVES / "F01_errors_seed0.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in infer:
        ax.scatter(r["latency"]["p95"], r["val_macro_f1"], s=65)
        ax.annotate(r["method"], (r["latency"]["p95"], r["val_macro_f1"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="p95 forward latency (ms, batch 1)", ylabel="Validation macro-F1", title="Inference accuracy and latency")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(CURVES / "inference_tradeoff.png", dpi=160)
    plt.close(fig)

    b_rows=[]
    for exp, name in BACKBONES:
        s=_summary(exp)
        if s:
            lat=backbone_latency.get(exp,{})
            b_rows.append((exp, name, s["weight_tag"], f"{s['params_m']:.2f}",
                           f"{s['gmacs_thop']:.2f}" if s["gmacs_thop"] is not None else "unavailable",
                           f"{s['val_macro_f1']:.4f}", f"{s['val_top1']:.4f}",
                           f"{s['mean_epoch_seconds']:.1f}",f"{lat['p95']:.2f}" if lat else "unavailable"))
    base = _summary("B01")
    t_rows=[]
    for exp, changes in TRAINING + ([('T12', json.loads((SUB/'T12_recipe.json').read_text())['changes'])] if (SUB/'T12_recipe.json').exists() else []):
        s=_summary(exp)
        if s:
            t_rows.append((exp, str(changes).replace('|','/'), f"{s['val_macro_f1']:.4f}",
                           f"{s['val_macro_f1']-base['val_macro_f1']:+.4f}"))
    i_rows=[(r["method"], r["k"], f"{r['val_macro_f1']:.4f}", f"{r['val_ece']:.4f}",
             f"{r['latency']['p50']:.2f}", f"{r['latency']['p95']:.2f}", f"{r['latency']['p99']:.2f}") for r in infer]
    f_rows=[]
    for exp, group in groups.items():
        for p, m in zip(group.preds, group.metrics):
            v=read_pred(str(PREDS/f"{exp}_seed{p.seed}_val.csv"))
            vm=compute_metrics(v.y_true,v.y_pred,v.probs)
            f_rows.append((exp, p.seed, f"{vm['macro_f1']:.4f}", f"{m['macro_f1']:.4f}",
                           f"{m['top1']:.4f}", f"{m['ece']:.4f}"))
        f_rows.append((exp,"mean ± std","—",fmt(*group.summary["macro_f1"]),
                       fmt(*group.summary["top1"]),fmt(*group.summary["ece"])))
    class_rows=[]
    for i, name in enumerate(CLASS_NAMES):
        class_rows.append((name, int(final.metrics[0]["support"][i]),
                           fmt(final.summary["precision"][0][i],final.summary["precision"][1][i],3),
                           fmt(final.summary["recall"][0][i],final.summary["recall"][1][i],3),
                           fmt(final.summary["f1"][0][i],final.summary["f1"][1][i],3)))
    delta=final.summary["macro_f1"][0]-baseline.summary["macro_f1"][0]
    noise=max(final.summary["macro_f1"][1],baseline.summary["macro_f1"][1])
    conclusion=("Chênh lệch lớn hơn độ lệch chuẩn lớn nhất giữa hai nhóm seed."
                if delta>noise else "Chênh lệch chưa vượt độ lệch chuẩn giữa các seed; chưa thể kết luận cải thiện ổn định.")
    pairs=cm.copy(); np.fill_diagonal(pairs,0)
    pair_indices=np.dstack(np.unravel_index(np.argsort(pairs.ravel())[::-1],pairs.shape))[0]
    top_pairs=[f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]}: {pairs[i,j]} ảnh" for i,j in pair_indices if pairs[i,j]>0][:5]
    source=locked["source"]
    best_backbone=max((exp for exp,_ in BACKBONES if _summary(exp)),
                      key=lambda exp:_summary(exp)["val_macro_f1"])
    best_training=max((exp for exp,_ in TRAINING if _summary(exp)),
                      key=lambda exp:_summary(exp)["val_macro_f1"])
    baseline_infer=next(r for r in infer if r["method"]=="I00")
    calibration=next((r for r in infer if r["method"]=="I04"),None)
    calibration_note=(f"F01 dùng {locked['method']}, không dùng temperature scaling trên test; vì vậy I4(a) của eval.py không được chấm. "
                      if locked["method"]!="I04" else "F01 dùng temperature scaling đã khớp trên val. ")
    backbone_gain=_summary(best_backbone)["val_macro_f1"]-base["val_macro_f1"]
    training_gain=_summary(best_training)["val_macro_f1"]-base["val_macro_f1"]
    inference_gain=selected["val_macro_f1"]-baseline_infer["val_macro_f1"]
    combo_path=SUB/"T12_recipe.json"
    if combo_path.exists() and _summary("T12"):
        combo=json.loads(combo_path.read_text())
        aug_source,loss_source=combo["augmentation_source"],combo["loss_source"]
        combo_delta=_summary("T12")["val_macro_f1"]-max(_summary(aug_source)["val_macro_f1"],_summary(loss_source)["val_macro_f1"])
        combo_text=(f"T12 kết hợp {aug_source} và {loss_source}: macro-F1 val {_summary('T12')['val_macro_f1']:.4f}, "
                    f"chênh {combo_delta:+.4f} so với thành phần đơn lẻ tốt hơn. "
                    "Đây là một seed; chưa thể khẳng định hai yếu tố cộng dồn nếu chênh lệch nhỏ.")
    else:
        combo_text="T12 chưa có kết quả để đánh giá tương tác giữa các yếu tố."
    class_counts=eda.get("all_classes",[])
    mismatch_count=len(eda.get("label_disagreements",[]))
    imbalance=(f"Lớp Negatives có {class_counts[8]:,} ảnh, chiếm {class_counts[8]/sum(class_counts):.1%}; "
               f"các lớp cây có {min(class_counts[:8]):,}–{max(class_counts[:8]):,} ảnh/lớp. "
               "Phân bố này khớp số lượng trong Table 1 của bài báo và giải thích vì sao báo cáo thêm macro-F1, recall từng lớp."
               if len(class_counts)==9 else "Phân bố lớp được thể hiện trong biểu đồ dưới đây.")
    sanity_text=(f"Kiểm pipeline: CE lý thuyết ln 9 = {sanity['uniform_ce_theory']:.4f}; "
                 f"loss head ban đầu {sanity['initial_head_loss']:.4f}, sau overfit một batch còn {sanity['overfit_final_loss']:.6f}; "
                 f"sai khác Focal γ=0 với CE là {sanity['focal_gamma0_error']:.2e}. "
                 "Ảnh kiểm augmentation và CutMix ở `curves/sanity_augmented.png` và `curves/sanity_cutmix.png`."
                 if sanity else "Ảnh kiểm augmentation và CutMix được lưu trong `curves/`.")
    robot=[]
    for budget in (30,100):
        feasible=[r for r in infer if r["latency"]["p95"]<=budget]
        if feasible:
            choice=max(feasible,key=lambda r:(r["val_macro_f1"],-r["val_ece"]))
            robot.append(f"Với giới hạn p95 {budget} ms trên thiết bị đo, {choice['method']} có macro-F1 val {choice['val_macro_f1']:.4f} và p95 {choice['latency']['p95']:.2f} ms.")
        else:
            robot.append(f"Không có phương pháp nào đạt giới hạn p95 {budget} ms trên thiết bị đo.")
    lines=[
        "# Báo cáo DeepWeeds Day 2", "",
        "## 1. Tóm tắt", "",
        f"Đã chạy 5 backbone, {len(t_rows)} công thức huấn luyện và {len(infer)} phương pháp suy luận trên fold 0. "
        f"Chọn {source} + {cfg['final_inference']} bằng validation, rồi huấn luyện lại F01 và T00 với ba seed. "
        f"F01 đạt macro-F1 test {fmt(*final.summary['macro_f1'])}, top-1 {fmt(*final.summary['top1'])}. "
        f"So với T00, Δ macro-F1 = {delta:+.4f}. {conclusion}", "",
        "## 2. Dữ liệu và thiết lập", "",
        f"DeepWeeds gồm {eda['split']['union']:,} ảnh, fold 0: train {eda['split']['n']['train']:,}, "
        f"val {eda['split']['n']['val']:,}, test {eda['split']['n']['test']:,}; giao giữa các tập bằng 0. "
        "Nhãn train/val/test lấy theo CSV fold 0 của tác giả. Chỉ dùng val để chọn mô hình, công thức, checkpoint và suy luận. "
        f"Test được đánh giá một lượt cho mỗi seed sau khi chốt lựa chọn. Có {mismatch_count} ảnh có nhãn khác giữa labels.csv và CSV fold 0; "
        "bài làm giữ nguyên nhãn của CSV fold 0.", "",
        f"Baseline: {baseline_cfg['backbone']}, ImageNet pretrained ({baseline_cfg['weight_tag']}), "
        f"{baseline_cfg['epochs']} epoch, batch {baseline_cfg['batch_size']}, ảnh {baseline_cfg['img_size']} px, "
        f"AdamW, LR backbone {baseline_cfg['lr_backbone']}, head {baseline_cfg['lr_head']}, "
        f"weight decay {baseline_cfg['weight_decay']}, warmup {baseline_cfg['warmup_epochs']} epoch rồi cosine; "
        f"seed 0/1/2. Thiết bị cuối: {gpu_name}; PyTorch {cfg['versions']['torch']}, timm {cfg['versions']['timm']}. "
        "Chỉ số tính bằng eval.py gốc; ECE dùng 15 bin; std mẫu ddof=1.", "",
        imbalance, "", sanity_text, "",
        "![Phân bố lớp](curves/class_distribution.png)", "",
        "![Ảnh mẫu từng lớp](curves/class_samples.png)", "",
        "## 3. So sánh backbone trên val", "",
        _table(["ID","Backbone","Tag","Params M","GMAC thop","Macro-F1 val","Top-1 val","s/epoch","p95 ms"],b_rows), "",
        f"Trong năm backbone, {best_backbone} có macro-F1 val cao nhất ({_summary(best_backbone)['val_macro_f1']:.4f}) ở seed 0.", "",
        "Cùng fold 0, seed 0 và công thức nền. GMAC do thop ước tính; xem riêng độ trễ đo trên GPU trong results.xlsx. "
        "Các chênh lệch ở bước sàng lọc này chỉ dựa trên một seed.", "",
        "## 4. Công thức huấn luyện trên val", "",
        _table(["ID","Thay đổi so với T00","Macro-F1 val","Δ với B01"],t_rows), "",
        f"Trong các công thức đơn lẻ, {best_training} có macro-F1 val cao nhất ({_summary(best_training)['val_macro_f1']:.4f}); "
        f"B01 đạt {_summary('B01')['val_macro_f1']:.4f}. Đây là so sánh seed 0.", "",
        "Mỗi T01–T11 thay một yếu tố so với B01/T00; T12 kết hợp augmentation và loss chọn bằng val. "
        "Các Δ này là kết quả một seed, chưa đủ để kết luận hiệu quả ổn định khi chênh lệch nhỏ.", "",
        combo_text, "",
        f"Đường cong B01 nằm ở `curves/B01_seed0.png`; checkpoint tốt nhất ở epoch {_summary('B01')['best_epoch']}. "
        f"Đường cong {best_training} nằm ở `curves/{best_training}_seed0.png`, checkpoint tốt nhất ở epoch {_summary(best_training)['best_epoch']}. "
        "So sánh loss train/val trên các đường cong để nhận diện hội tụ và quá khớp; giá trị từng epoch nằm trong log.", "",
        "## 5. Suy luận và chi phí", "",
        _table(["ID","Views","Macro-F1 val","ECE val","p50 ms","p95 ms","p99 ms"],i_rows), "",
        f"Các phép đo trên {selected['device']}, batch 1, FP32, 10 warmup và 100 lần đo, đồng bộ CUDA khi dùng GPU; "
        "phạm vi chỉ gồm forward với đầu vào tổng hợp. Thông lượng và độ trễ theo batch khác có trong results.xlsx. "
        f"Chọn {cfg['final_inference']} trên val: macro-F1 {selected['val_macro_f1']:.4f}, "
        f"ECE {selected['val_ece']:.4f}, p95 {selected['latency']['p95']:.2f} ms. "
        +
        (f"Temperature scaling đổi ECE val từ {baseline_infer['val_ece']:.4f} sang {calibration['val_ece']:.4f}; "
         f"macro-F1 đổi từ {baseline_infer['val_macro_f1']:.4f} sang {calibration['val_macro_f1']:.4f}. " if calibration else "")
        + calibration_note + " ".join(robot) + " Chỉ đo forward; cần đo cả pipeline trước khi chọn phương án triển khai thực tế.", "",
        "![Đánh đổi độ trễ và macro-F1](curves/inference_tradeoff.png)", "",
        "## 6. Chung kết trên test", "",
        f"Cấu hình F01: `{json.dumps({k:cfg[k] for k in ('backbone','init','aug','loss','sampler','mix','ema_decay','epochs','batch_size','img_size','lr_backbone','lr_head','weight_decay','final_inference')},ensure_ascii=False)}`.", "",
        _table(["ID","Seed","Macro-F1 val","Macro-F1 test","Top-1 test","ECE test"],f_rows), "",
        f"Mốc T00 macro-F1 test {fmt(*baseline.summary['macro_f1'])}; F01 cải thiện {delta:+.4f}. {conclusion}", "",
        _table(["Lớp","Ảnh test","Precision F01","Recall F01","F1 F01"],class_rows), "",
        "Ma trận nhầm lẫn dưới đây thuộc F01 seed 0 (hàng là nhãn thật, cột là nhãn dự đoán). "
        "Các cặp nhầm nhiều nhất của seed này: " + ("; ".join(top_pairs) if top_pairs else "không có") + ". "
        f"Riêng Chinee Apple → Snake Weed: {cm[0,7]} ảnh; Snake Weed → Chinee Apple: {cm[7,0]} ảnh. "
        "Giả thuyết cần kiểm thêm: hình dạng lá, nền và điều kiện chiếu sáng tương tự có thể làm hai lớp khó phân biệt; "
        "ảnh lỗi bên dưới chỉ minh họa, chưa chứng minh nguyên nhân.", "",
        "![Confusion matrix](curves/F01_confusion_seed0.png)", "",
        "Ảnh dưới đây là tối đa 12 lỗi test có độ tin cậy dự đoán sai cao nhất của F01 seed 0. "
        "Chúng minh họa kiểu lỗi; không dùng để chỉnh mô hình sau khi xem test.", "",
    ]
    if errors:
        lines += ["![Lỗi dự đoán](curves/F01_errors_seed0.png)", ""]
    else:
        lines += ["Seed 0 không có dự đoán sai trên test.", ""]
    lines += ["## 7. Kết luận và hạn chế", "",
              f"Cấu hình được chọn bằng val là {source} + {cfg['final_inference']}. "
              f"Mức cải thiện test so với baseline là {delta:+.4f} macro-F1. {conclusion} "
              f"Trên val seed 0, backbone cao nhất hơn B01 {backbone_gain:+.4f}, công thức đơn lẻ cao nhất hơn B01 {training_gain:+.4f}, "
              f"và phương pháp suy luận đã chọn đổi macro-F1 so với I00 trên cùng nguồn {inference_gain:+.4f}. "
              "Ba chênh lệch này thuộc các phép sàng lọc khác nhau, không cộng lại thành đóng góp nhân quả. "
              "Không dùng test để chọn lại cấu hình.", "",
              "Nghiên cứu chỉ dùng fold 0 và ba seed ở chung kết. Các thí nghiệm sàng lọc dùng một seed. "
              "Fold của DeepWeeds không tách theo địa điểm hoặc mùa, nên điểm test có thể lạc quan khi triển khai ngoài miền dữ liệu. "
              "Độ trễ forward với ảnh tổng hợp chưa bao gồm đọc ảnh, tiền xử lý hay truyền dữ liệu; cần đo toàn pipeline trên robot trước khi triển khai.", "",
              "## 8. Tái lập", "",
              "Cấu hình, tag trọng số, phiên bản và lịch sử epoch nằm trong `logs/<exp_id>/seed<k>/` của bài nộp; "
              "checkpoint được giữ tại `runs/<exp_id>/seed<k>/` trên server, không đưa lên Git. "
              "File dự đoán `predictions/F01_seed*_test.csv` và `predictions/T00_seed*_test.csv` là đầu vào trực tiếp cho `eval.py score`/`grade`. "
              "Notebook: `code/lab_day2.ipynb`; link Colab trực tiếp nằm trong README. "
              "Danh sách tất cả cấu hình có trong `results.xlsx` và `logs/`; đầu ra `eval.py score`/`grade` nằm trong `evaluation/`.", ""]
    path=SUB/"report.md"
    path.write_text("\n".join(lines),encoding="utf-8")
    print("Wrote",path)
    return path


if __name__ == "__main__":
    generate_report()
