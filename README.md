# Solution Path Pruning

Official implementation of **"Adaptive Pruning of Pretrained Transformer via Differential Inclusions"**, published at **ICLR 2025**.

SPP is a novel pruning framework that enables **adaptive compression** of pretrained transformers. Instead of pruning at a fixed compression ratio, SPP constructs a **Transformer Weight Family** with different sparsity levels in a single search stage. This allows users to efficiently deploy transformers with varying levels of sparsity without redoing the pruning process for each ratio.

---

## **Features**

- 📉 **Adaptive pruning**: Generates pruned models with different sparsity levels in a single pruning process.
- ⚡ **Efficient compression**: Reduces computational costs while preserving model accuracy.
- 🔍 **Pair-wise structured pruning**: Maintains the transformer’s functional structure while enabling flexibility in compression.
- 🏆 **State-of-the-art performance**: Outperforms existing pruning methods on **DeiT, Swin, CLIP, and LLMs (Llama2, OPT)**.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/yizhuoDi/Solution-Path-Pruning.git
cd Solution-Path-Pruning
```

### **2. Create a Virtual Environment and Install Dependencies**
```bash
conda env create -n spp -f spp.yml
source activate spp
```

---

## **Usage**

### **1. Run Pruning on Pretrained Transformers**

SPP supports pruning on multiple transformer architectures, such as DeiT.

#### **Search the Sparse model on ImageNet-1k**
```bash
sh deit_search.sh
```

### **2. Finetune Pruned Model**

After pruning, you can fine-tune the model to recover accuracy:
```bash
sh deit_retrain.sh
```

---

## **Main Results**

We evaluated **SPP** on multiple datasets and model architectures, demonstrating its **superior performance** over traditional pruning methods.

| Model | Dataset | Params Reduction | Accuracy (%) |
|--------|--------|-----------------|--------------|
| DeiT-Small | ImageNet-1k | 70.6% | **80.2** |
| Swin-Tiny | ImageNet-1k | 66.1% | **80.6** |
| CLIP-Large | COCO | 75.9% | **70.8** (Image-to-Text) |
| Llama2-7B | ARC-e | 50.0% | **71.8** |

More results and comparisons can be found in our [paper](https://arxiv.org/pdf/2501.03289).

---

## **Citation**

If you find our work useful, please cite:

```bibtex
@article{ding2025adaptive,
  title={Adaptive Pruning of Pretrained Transformer via Differential Inclusions},
  author={Ding, Yizhuo and Fan, Ke and Wang, Yikai and Sun, Xinwei and Fu, Yanwei},
  journal={arXiv preprint arXiv:2501.03289},
  year={2025}
}
```

---

## **License**

This project is released under the **MIT License**.

---

## **Contact**

For questions, please open an issue on GitHub or contact us via email at **yizhuo.ding@example.com**.

