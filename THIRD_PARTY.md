# Third-party components

- Bundle-MLLM: *Fine-tuning Multimodal Large Language Models for Product Bundling*,
  KDD 2025, https://arxiv.org/abs/2407.11712. The model, feature-loading and prompt
  implementation includes adapted code from this project; its public repository
  uses MIT. The upstream copyright and license are included in
  `licenses/Bundle-MLLM.txt`. Source: https://github.com/Xiaohao-Liu/Bundle-MLLM.
- Dataset/features: https://huggingface.co/datasets/xhLiu/BundleConstruction.
  Keep the original dataset terms; the code's license does not automatically
  apply to data or media.
- Llama-2: https://huggingface.co/meta-llama/Llama-2-7b-hf. Obtain the backbone
  directly under its license; this package contains adaptation weights only.
- Rebuilt LightGCN relationships follow the implementation equations at
  https://github.com/gusye1234/LightGCN-PyTorch. The fitting source identifies this
  reference and does not claim to recover the unknown upstream vector recipe.

No new license is assigned to third-party assets. The authors must choose a
license for their own modifications before public release.
