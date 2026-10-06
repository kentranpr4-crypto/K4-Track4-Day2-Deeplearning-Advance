# DeepWeeds Lab Day 2 - Bao cao ket qua

## Trang thai

Bao cao duoc tao tu log va prediction that; cac muc chua co so lieu can duoc hoan thien sau khi chay Kaggle.

## Backbone tren validation

| exp_id | backbone | seed | macro_f1_val | top1_val | best_epoch | params_m | gmac | latency_batch1_ms | train_seconds_per_epoch | pretrained_tag | img_size | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B01 | resnet50 | 0 | 0.8712313173670629 | 0.9031705227077977 | 11 | 23.526473 | 4.131713024 | 7.214563001980423 | 145.73625930899993 | timm/resnet50.a1_in1k | 224 | 12 |
| B02 | resnext50_32x4d | 0 | 0.8934738323413968 | 0.9180234218794631 | 11 | 22.998345 | 4.286154752 | 8.419552499617566 | 142.53274914066662 | timm/resnext50_32x4d.a1h_in1k | 224 | 12 |
| B03 | convnext_tiny | 0 | 0.9675194506742736 | 0.9745786918023421 | 11 | 27.827049 | 4.454808576 | 6.10910000068543 | 321.95159971083314 | timm/convnext_tiny.in12k_ft_in1k | 224 | 12 |
| B04 | vit_small_patch16_224 | 0 | 0.9458499360621921 | 0.9597257926306769 | 12 | 21.669129 | 4.240838016 | 5.7550144993001595 | 369.67207315783327 | timm/vit_small_patch16_224.augreg_in21k_ft_in1k | 224 | 12 |
| B05 | efficientnet_b0 | 0 | 0.8746837328609423 | 0.9031705227077977 | 11 | 4.019077 | 0.384610272 | 8.12610550201498 | 367.2788949341663 | timm/efficientnet_b0.ra_in1k | 224 | 12 |

## Ablation huan luyen

| exp_id | backbone | seed | macro_f1_val | top1_val | best_epoch | params_m | gmac | latency_batch1_ms | train_seconds_per_epoch | init | aug | loss | mix | sampler | ema_decay | delta_macro_f1_vs_B03 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | convnext_tiny | 0 | 0.8656963341817592 | 0.8908883176235362 | 8 | 27.827049 | None | None | 93.66681196083334 | frozen | basic | ce | None | None | None | -0.10182311649251441 |
| T02 | convnext_tiny | 0 | 0.9632944132792739 | 0.9700085689802913 | 11 | 27.827049 | None | None | 252.5959650525834 | finetune | color | ce | None | None | None | -0.004225037394999687 |
| T03 | convnext_tiny | 0 | 0.961665075767857 | 0.9700085689802913 | 8 | 27.827049 | None | None | 159.86444680825025 | finetune | randaug | ce | None | None | None | -0.005854374906416604 |
| T04 | convnext_tiny | 0 | 0.9614439386240947 | 0.9714367323621822 | 11 | 27.827049 | None | None | 161.16550750208316 | finetune | basic | ls | None | None | None | -0.006075512050178866 |
| T05 | convnext_tiny | 0 | 0.9645034007962335 | 0.9717223650385605 | 12 | 27.827049 | None | None | 135.34204516925016 | finetune | basic | focal | None | None | None | -0.00301604987804005 |
| T06 | convnext_tiny | 0 | 0.9610835820742208 | 0.9700085689802913 | 11 | 27.827049 | None | None | 131.4183403762504 | finetune | basic | ce_weighted | None | None | None | -0.006435868600052741 |
| T07 | convnext_tiny | 0 | 0.9550207986181307 | 0.9642959154527277 | 12 | 27.827049 | None | None | 223.31406129975008 | finetune | color | focal | None | None | None | -0.012498652056142867 |

## Phuong phap suy luan

Chua co ket qua.

## Ket qua test

Chua co ket qua.

## Chi so tung lop

Chua co ket qua.

## Phan tich va han che can bo sung

- Giai thich moi thay doi dua tren ket qua validation va do nhieu qua seed.
- Them ma tran nham lan, anh du doan sai va khuyen nghi latency cho robot.
- Neu gioi han: mot fold, chia ngau nhien theo anh thay vi dia diem, co the lac quan tren dia diem moi.
