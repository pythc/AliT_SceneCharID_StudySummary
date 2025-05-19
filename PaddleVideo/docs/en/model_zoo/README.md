[简体中文](../../zh-CN/model_zoo/README.md) | English

# Introduction

We implemented video classification model and action localization model in this repo.

## Model list

| Field | Model | Config | Dataset | Metrics | ACC% | Download |
| :--------------- | :--------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| action recgonition | [ppTSM](./recognition/pp-tsm.md) | [pptsm.yaml](../../../configs/recognition/pptsm/pptsm_k400_frames_dense.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 76.16 | [ppTSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense_distill.pdparams) |
| action recgonition | [SlowFast](./recognition/slowfast.md) | [slowfast_multigrid.yaml](../../../configs/recognition/slowfast/slowfast_multigrid.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 75.84 | [SlowFast.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast_8*8.pdparams) |
| action recgonition | [TSM](./recognition/tsm.md) | [tsm.yaml](../../../configs/recognition/tsm/tsm_k400_frames.yaml)  | [Kinetics-400](../dataset/k400.md) | Top-1 | 70.86 | [TSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams) |
| action recgonition | [TSN](./recognition/tsn.md) | [tsn.yaml](../../../configs/recognition/tsn/tsn.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 67.0 | TODO |
| action recgonition | [AttentionLSTM](./recognition/attention_lstm.md) | [attention_lstm.yaml](../../../configs/recognition/attention_lstm/attention_lstm.yaml) | [Youtube-8M](../dataset/youtube8m.md) | Hit@1 | 89.0 | [AttentionLstm.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/AttentionLstm/AttentionLstm.pdparams) |
| action detection| [BMN](./localization/bmn.md) | [bmn.yaml](../../../configs/localization/bmn.yaml) | [ActivityNet](../dataset/ActivityNet.md) |  AUC | 67.23 | [BMN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/BMN/BMN.pdparams) |


# Reference

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen

- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al. 

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool

- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1), Ji Lin, Chuang Gan, Song Han
