## 環境需求

*   **硬體:**
    *   **NVIDIA GPU:** 一張支援 CUDA 的 NVIDIA 顯示卡。
    *   **RAM:** 取決於所選模型大小。
    *   **儲存空間:** 存模型快取。
*   **軟體:**
    *   **Python:** 建議版本 3.12.6+
    *   **Anaconda:** 用於管理 Python 環境和依賴。
    *   **NVIDIA 驅動程式:** 最新的穩定版 NVIDIA 顯示卡驅動程式。
    *   **CUDA Toolkit:** 與 PyTorch 版本相容的 CUDA Toolkit 版本。

## 安裝步驟

1.  **Clone 專案庫:**
    ```bash
    git clone https://github.com/flyflyk/LLM_Offload.git
    cd LLM_Offload
    ```

2.  **(推薦) 建立並啟用 Conda 環境:**
    ```bash
    conda create -n llm_inference_env python=3.12.6 -y
    conda activate llm_inference_env
    ```

3.  **安裝 PyTorch (GPU 版本):**
    *   前往 [PyTorch 官方網站 Get Started 頁面](https://pytorch.org/get-started/locally/)。

4.  **安裝其他依賴套件:**
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

1.  **配置文件:**
    在 `config.py` 中，修改需要的參數。

2.  **執行主腳本:**
    ```bash
    python main.py
    ```